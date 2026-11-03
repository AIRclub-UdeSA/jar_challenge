"""Contrato ROS: mapa, calidad, frescura, TF, QoS y presupuesto de reportes."""
import json
import math
import time

import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String

from equipo_jar.nodo_percepcion import NodoPercepcion
from test_percepcion import caras


class Captura:
    def __init__(self): self.mensajes=[]
    def publish(self,msg): self.mensajes.append(msg)


@pytest.fixture
def node():
    rclpy.init()
    node=NodoPercepcion(parameter_overrides=[Parameter('cantidad_victimas',value=1)])
    node.pub_reportes=Captura();node.pub_estado=Captura()
    yield node
    node.destroy_node();rclpy.shutdown()


def preparar(node):
    msg=OccupancyGrid()
    msg.header.frame_id='map';msg.info.width=100;msg.info.height=100
    msg.info.resolution=.1;msg.info.origin.position.x=-4.;msg.info.origin.position.y=-4.
    msg.info.origin.orientation.w=1.;msg.data=[0]*10000
    node.al_mapa(msg)
    node.al_localizacion(String(data=json.dumps(dict(utilizable=True,std_xy_m=.01,std_yaw_rad=.01))))


def nube(node,offset=0.):
    xyz,rgb=caras()
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
    rows=np.zeros(len(xyz),dtype)
    for i,name in enumerate(('x','y','z')):rows[name]=xyz[:,i]
    rows['rgb']=(rgb[:,0].astype(np.uint32)<<16)|(rgb[:,1].astype(np.uint32)<<8)|rgb[:,2]
    msg=PointCloud2()
    msg.header.frame_id='depth_test'
    msg.header.stamp=node.get_clock().now().to_msg()
    msg.header.stamp.sec-=int(offset)
    msg.width=len(rows);msg.height=1;msg.point_step=16;msg.row_step=16*len(rows)
    msg.fields=[PointField(name=n,offset=4*i,datatype=7,count=1)
                for i,n in enumerate(('x','y','z','rgb'))]
    msg.data=rows.tobytes()
    tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id='depth_test'
    tf.transform.rotation.w=1.
    node.tf_buffer.set_transform_static(tf,'test')
    return msg


def test_sin_mapa_o_localizacion_no_reporta(node):
    node.al_nube(nube(node))
    assert not node.pub_reportes.mensajes
    assert not node.disponible()


def test_localizacion_vencida_no_admite_datos(node):
    preparar(node);node.t_localizacion=time.monotonic()-5
    node.al_nube(nube(node))
    assert not node.registro.victimas


def test_nube_vencida_no_confirma(node):
    preparar(node);node.al_nube(nube(node,offset=3))
    assert not node.registro.victimas
    assert json.loads(node.pub_estado.mensajes[-1].data)['motivo']=='NUBE_VENCIDA'


def test_tf_ausente_se_abstiene(node):
    preparar(node);msg=nube(node);msg.header.frame_id='inexistente'
    node.al_nube(msg)
    assert not node.registro.victimas


def test_tf_tardio_reintenta_sin_bloquear_y_sin_cambiar_estampa(node):
    preparar(node);msg=nube(node);msg.header.frame_id='tardio'
    node.al_nube(msg)
    assert node.nube_pendiente is msg
    tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id='tardio'
    tf.transform.rotation.w=1.;node.tf_buffer.set_transform_static(tf,'test')
    node.cerrar_rafaga()
    assert node.nube_pendiente is None
    assert len(node.registro.victimas)==1


def test_espera_tf_no_recicla_nube_vencida(node):
    preparar(node);msg=nube(node,offset=4)
    node.nube_pendiente=msg;node.cerrar_rafaga()
    assert node.nube_pendiente is None
    assert not node.registro.victimas


def test_reporte_valido_unico_y_presupuesto(node):
    preparar(node)
    for _ in range(5):
        node.ultima_nube=-math.inf
        node.al_nube(nube(node))
        time.sleep(.42)
        node.t_localizacion=time.monotonic()
    assert len(node.pub_reportes.mensajes)==1
    msg=node.pub_reportes.mensajes[0]
    assert msg.header.frame_id=='map' and msg.point.z==0.
    assert math.dist((msg.point.x,msg.point.y),(1.,.2))<.025
    assert not node.disponible()  # N agotado


def test_imagen_roja_abre_rafaga_best_effort_y_cierra(node):
    preparar(node)
    msg=Image();msg.encoding='rgb8';msg.width=10;msg.height=10;msg.step=30
    msg.data=bytes([220,0,0]*100);msg.header.stamp=node.get_clock().now().to_msg()
    node.al_imagen(msg)
    assert node.sub_nube is not None
    assert node.sub_nube.qos_profile.reliability==ReliabilityPolicy.BEST_EFFORT
    node.fin_rafaga=-math.inf;node.cerrar_rafaga()
    assert node.sub_nube is None


def test_azul_no_abre_nube(node):
    preparar(node)
    msg=Image();msg.encoding='rgb8';msg.width=10;msg.height=10;msg.step=30
    msg.data=bytes([0,0,220]*100);msg.header.stamp=node.get_clock().now().to_msg()
    node.al_imagen(msg)
    assert node.sub_nube is None

def test_cara_incompleta_expone_vistas_pero_no_publica_reporte(node):
    preparar(node)
    msg=nube(node)
    msg.width=720;msg.row_step=16*720;msg.data=msg.data[:msg.row_step]
    node.al_nube(msg)
    status=json.loads(node.pub_estado.mensajes[-1].data)
    assert status['necesita_otra_vista']
    assert status['vistas_sugeridas']
    assert not node.pub_reportes.mensajes
    assert node.registro.reportes==0


def preparar_rgb_tardio(node, offset=0):
    preparar(node)
    node.pub_observacion = Captura()
    info = CameraInfo()
    info.header.frame_id = 'rgb_tardio'
    info.width = 640
    info.height = 480
    info.k = [500., 0., 320., 0., 500., 240., 0., 0., 1.]
    node.al_info_camara(info)
    msg = Image()
    msg.header.frame_id = info.header.frame_id
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.stamp.sec -= offset
    return msg


def test_rgb_tf_tardio_reintenta_estampa_original(node):
    msg = preparar_rgb_tardio(node)
    node.publicar_observacion(msg.header, False)
    assert node.observacion_pendiente == (msg.header, False)
    assert not node.pub_observacion.mensajes
    tf = TransformStamped()
    tf.header.frame_id = 'map'
    tf.child_frame_id = msg.header.frame_id
    tf.transform.rotation.y = math.sin(math.pi/4)
    tf.transform.rotation.w = math.cos(math.pi/4)
    node.tf_buffer.set_transform_static(tf, 'test')
    node.cerrar_rafaga()
    assert node.observacion_pendiente is None
    rows = node.pub_observacion.mensajes
    assert len(rows) == 1
    actual = json.loads(rows[0].data)
    assert actual['estampa_s'] == msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
    assert abs(actual['pose_camara'][2]) < 1e-9
    node.cerrar_rafaga()
    assert len(rows) == 1


@pytest.mark.parametrize('terminada', [False, True])
def test_rgb_pendiente_no_recicla_vencidos_ni_despues_done(node, terminada):
    msg = preparar_rgb_tardio(node, offset=4 if not terminada else 0)
    node.observacion_pendiente = (msg.header, True)
    node.terminada = terminada
    node.cerrar_rafaga()
    assert node.observacion_pendiente is None
    assert not node.pub_observacion.mensajes


def test_observacion_rgb_incluye_rumbo_rojo_sin_necesitar_nube(node):
    preparar(node);node.pub_observacion=Captura()
    info=CameraInfo();info.header.frame_id='rgb_optico';info.width=320;info.height=240
    info.k=[250.,0.,160.,0.,250.,120.,0.,0.,1.];node.al_info_camara(info)
    tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id='rgb_optico'
    tf.transform.rotation.x=-.5;tf.transform.rotation.y=.5
    tf.transform.rotation.z=-.5;tf.transform.rotation.w=.5
    node.tf_buffer.set_transform_static(tf,'test')
    pixels=np.zeros((240,320,3),np.uint8);pixels[70:120,240:270]=[220,0,0]
    msg=Image();msg.header.frame_id='rgb_optico';msg.header.stamp=node.get_clock().now().to_msg()
    msg.width=320;msg.height=240;msg.encoding='rgb8';msg.step=960;msg.data=pixels.tobytes()
    node.al_imagen(msg)
    d=json.loads(node.pub_observacion.mensajes[-1].data)
    assert d['region_roja'] and len(d['regiones_rojas'])==1
    assert d['regiones_rojas'][0]['desvio_rad']<-.3 # derecha en imagen => giro horario
    assert not node.pub_reportes.mensajes
