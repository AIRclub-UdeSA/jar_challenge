"""Small M4 mission over workshop Nav2; existing M1 and M3 unchanged.

This node is the only /cmd_vel writer. Localization commands have authority
until usable; then fresh Nav2 commands are forwarded. Stop on stale sensors,
localization or camera; never report /done away from the captured start.
"""
import json
import math
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.clock import Clock, ClockType
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Empty
from tf2_ros import Buffer, TransformListener, TransformException
from equipo_jar.navegacion.grilla import grilla_desde_occupancy
from equipo_jar.cobertura_visual import CoberturaVisual
from equipo_jar.mision import clave_vista
from equipo_jar.mision_simple import siguiente_vista

RETAINED=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,
                   durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


class MisionNav2(Node):
    def __init__(self):
        super().__init__('mision_nav2')
        self.declare_parameter('cantidad_victimas',5)
        self.declare_parameter('duracion_s',600.)
        self.declare_parameter('automatico',True)
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.client=ActionClient(self,NavigateToPose,'navigate_to_pose')
        self.pub=self.create_publisher(Twist,'/cmd_vel',10)
        self.status=self.create_publisher(String,'/equipo/mision_estado',RETAINED)
        self.done_pub=self.create_publisher(Empty,'/done',10)
        self.grid=self.coverage=self.costs=None
        self.start=None;self.begin=None;self.view=None
        self.state='ESPERANDO_LOCALIZACION';self.reason=''
        self.local={};self.local_at=-math.inf;self.camera_at=-math.inf
        self.scan_at=-math.inf;self.commands={}
        self.attempted=set();self.suggested=[];self.attempt_counts={}
        self.reports=0;self.reported=[];self.extra=None;self.extrinsic=None
        self.goal=None;self.sending=False;self.result=None;self.canceling=False
        self.returning=False;self.done=False;self.arrived_at=None;self.next_select=0.
        self.goal_at=None
        self.return_distance=0.
        self.goal_return=False
        self.selection=None
        self.worker=ThreadPoolExecutor(max_workers=1)
        self.create_subscription(OccupancyGrid,'/map',self.map_seen,RETAINED)
        self.create_subscription(OccupancyGrid,'/global_costmap/costmap',self.costmap_seen,RETAINED)
        self.create_subscription(String,'/equipo/localizacion_estado',self.local_seen,RETAINED)
        self.create_subscription(String,'/equipo/percepcion_estado',self.perception,RETAINED)
        self.create_subscription(String,'/equipo/observacion_visual',self.visual,10)
        self.create_subscription(PoseStamped,'/equipo/meta_nav2',self.external_goal,10)
        self.create_subscription(LaserScan,'/scan',lambda m:setattr(self,'scan_at',time.monotonic()),qos_profile_sensor_data)
        for topic,key in (('/equipo/cmd_nav2','nav'),('/equipo/cmd_localizacion','local')):
            self.create_subscription(Twist,topic,lambda m,k=key:self.commands.update({k:(time.monotonic(),m)}),10)
        self.timer=self.create_timer(.1,self.tick,clock=Clock(clock_type=ClockType.STEADY_TIME))

    def clock_s(self):
        return self.get_clock().now().nanoseconds/1e9

    def external_goal(self,msg):
        # Diagnostic equivalent of M2's external-goal mode. No fixture files
        # are ever read by the strategy; disabled in autonomous mission mode.
        if (self.get_parameter('automatico').value or self.grid is None or
                not self.local.get('utilizable') or msg.header.frame_id!='map' or
                self.goal is not None or self.sending):return
        p=msg.pose.position;heading=yaw(msg.pose.orientation)
        if all(math.isfinite(v) for v in (p.x,p.y,heading)):
            self.send(dict(x=p.x,y=p.y,yaw=heading))

    def pose(self,stamp=None):
        t=self.tf.lookup_transform('map','base_footprint',stamp or Time())
        return t.transform.translation.x,t.transform.translation.y,yaw(t.transform.rotation)

    def map_seen(self,msg):
        if self.grid is not None or msg.header.frame_id!='map':return
        i=msg.info
        self.grid=grilla_desde_occupancy(msg.data,i.width,i.height,i.resolution,
            i.origin.position.x,i.origin.position.y,yaw(i.origin.orientation))
        self.coverage=CoberturaVisual(self.grid)
        self.begin=self.clock_s()

    def costmap_seen(self,msg):
        if self.grid is None or msg.header.frame_id!='map':return
        i=msg.info;g=self.grid
        if (i.width,i.height)!=(g.ancho,g.alto) or abs(i.resolution-g.resolucion)>1e-6:return
        if math.hypot(i.origin.position.x-g.origen_x,i.origin.position.y-g.origen_y)>1e-5:return
        data=np.asarray(msg.data).reshape(g.alto,g.ancho)
        self.costs=np.where((data>=0)&(data<99)&~g.ocupada,data/20.,np.inf)
        self.extra=(data==100)&~g.ocupada

    def local_seen(self,msg):
        self.local=json.loads(msg.data);self.local_at=time.monotonic()

    def perception(self,msg):
        d=json.loads(msg.data);self.reports=max(self.reports,d.get('reportes',0))
        self.reported=d.get('reportadas',self.reported)
        self.suggested=d.get('vistas_sugeridas',[])

    def visual(self,msg):
        if self.coverage is None or not self.local.get('utilizable'):return
        d=json.loads(msg.data);stamp=float(d['estampa_s'])
        if not 0<=self.clock_s()-stamp<1.:return
        try:p=self.pose(Time(seconds=stamp))
        except TransformException:return
        camera=d['pose_camara'];dx,dy=camera[0]-p[0],camera[1]-p[1]
        c,s=math.cos(p[2]),math.sin(p[2])
        self.extrinsic=(c*dx+s*dy,-s*dx+c*dy,math.atan2(math.sin(camera[2]-p[2]),math.cos(camera[2]-p[2])))
        self.coverage.p=replace(self.coverage.p,campo_horizontal_rad=d['campo_horizontal_rad'])
        self.coverage.registrar(camera,stamp,True,self.extra)
        self.camera_at=time.monotonic()

    def send(self,view):
        request=NavigateToPose.Goal();request.pose.header.frame_id='map'
        request.pose.header.stamp=self.get_clock().now().to_msg()
        request.pose.pose.position.x=float(view['x']);request.pose.pose.position.y=float(view['y'])
        request.pose.pose.orientation.z=math.sin(view['yaw']/2)
        request.pose.pose.orientation.w=math.cos(view['yaw']/2)
        self.view=view;self.sending=True;self.goal_at=self.clock_s()
        self.goal_return=self.returning
        if not self.returning:
            self.return_distance=float(view.get('regreso_m',0.))+float(view.get('camino_m',0.))
        self.client.send_goal_async(request).add_done_callback(self.accepted)
        self.state='REGRESANDO' if self.returning else 'YENDO_VISTA'
        self.get_logger().info(f'{self.state}: {json.dumps(view)}')

    def accepted(self,future):
        self.sending=False
        try:handle=future.result()
        except Exception as error:
            self.reason=str(error);self.finish_goal(False);return
        if not handle.accepted:self.finish_goal(False);return
        self.goal=handle;handle.get_result_async().add_done_callback(self.completed)

    def completed(self,future):
        try:success=future.result().status==4
        except Exception:success=False
        self.goal=None;self.canceling=False;self.finish_goal(success)

    def finish_goal(self,success):
        if self.view is not None:self.attempted.add(clave_vista(tuple(self.view[k] for k in ('x','y','yaw'))))
        self.state='OBSERVANDO' if success else 'BUSCANDO_VISTA'
        self.arrived_at=self.clock_s() if success else None
        self.next_select=time.monotonic()+.5
        self.reason='' if success else 'Nav2 no completó la meta; probar otra vista'

    def tick(self):
        now=time.monotonic();clock=self.clock_s()
        usable=self.local.get('utilizable',False) and now-self.local_at<1.
        sensors=now-self.scan_at<.8
        command=Twist()
        if self.grid is not None and sensors and not self.done:
            key='nav' if usable and now-self.camera_at<3. else 'local'
            at,proposed=self.commands.get(key,(-math.inf,Twist()))
            if now-at<.5 and (key=='local' or self.goal is not None):command=proposed
        self.pub.publish(command)
        if not usable or not sensors or self.grid is None or self.costs is None or self.extrinsic is None:
            self.publish_status();return
        try:p=self.pose()
        except TransformException:self.publish_status();return
        if self.start is None:self.start=p
        if not self.get_parameter('automatico').value:
            self.publish_status();return
        remaining=self.get_parameter('duracion_s').value-(clock-self.begin)
        should_return=(self.reports>=self.get_parameter('cantidad_victimas').value or
                       remaining<45.+max(self.return_distance,math.dist(p[:2],self.start[:2]))/.08)
        if should_return and not self.returning:
            self.returning=True
        if self.returning and not self.goal_return and self.goal is not None and not self.canceling:
            self.canceling=True;self.goal.cancel_goal_async()
        if self.done:self.publish_status();return
        if self.goal is not None or self.sending:
            limit=30.+float(self.view.get('camino_m',0.))/.1
            if clock-self.goal_at>limit and self.goal is not None and not self.canceling:
                self.canceling=True;self.goal.cancel_goal_async()
            self.publish_status();return
        if self.state=='OBSERVANDO' and clock-self.arrived_at<3.:
            self.publish_status();return
        if self.returning and math.dist(p[:2],self.start[:2])<.09:
            self.pub.publish(Twist());self.done_pub.publish(Empty());self.done=True
            self.state='FINALIZADA';self.publish_status();return
        if now<self.next_select or now-self.camera_at>3. or not self.client.server_is_ready():
            self.publish_status();return
        if self.returning:
            self.send(dict(zip(('x','y','yaw'),self.start),camino_m=self.return_distance))
        else:
            if self.selection is not None:
                if not self.selection.done():
                    self.publish_status();return
                try:view=self.selection.result()
                except Exception as error:
                    self.reason='Error de selección: '+str(error)
                    self.returning=True;self.selection=None
                    self.publish_status();return
                self.selection=None
                if view is None:
                    self.returning=True;self.reason='Sin más vistas alcanzables; búsqueda incompleta si N pendiente'
                else:self.send(view)
                self.publish_status();return
            suggested=[v for v in self.suggested if all(k in v for k in ('x','y','yaw')) and
                       not any(math.hypot(v.get('mirar_x',v['x'])-r['x'],v.get('mirar_y',v['y'])-r['y'])<.3 for r in self.reported)]
            self.selection=self.worker.submit(siguiente_vista,self.grid,self.costs.copy(),p,
                copy.deepcopy(self.coverage),set(self.attempted),self.extrinsic,copy.deepcopy(suggested),
                extra=self.extra.copy() if self.extra is not None else None,largada=self.start,
                restante_s=remaining)
            self.state='BUSCANDO_VISTA'
        self.publish_status()

    def publish_status(self):
        self.status.publish(String(data=json.dumps(dict(estado=self.state,motivo=self.reason,
            reportes=self.reports,done=self.done,vista=self.view,
            cobertura_observada=int(self.coverage.observada.sum()) if self.coverage else 0,
            backend='nav2_workshop'))))


def main(args=None):
    rclpy.init(args=args);node=MisionNav2()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.pub.publish(Twist());node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
