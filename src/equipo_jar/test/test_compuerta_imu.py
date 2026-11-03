import math
from types import SimpleNamespace as N
import pytest
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from rclpy.serialization import serialize_message
from equipo_jar.odometria_imu import RumboImu,OdometriaImu

def train(slope):
 f=RumboImu()
 for k in range(40):
  t=.1*k;f.agregar_imu(t,slope*.05*k);f.agregar_ruedas(t,.002*k,-.003*k,.05*k)
 return f

def test_102_pasa_ruedas_sin_reintegrar_pose():
 f=train(1.02)
 assert f.verificado and not f.correccion_necesaria and not f.usando_imu
 assert f.pose==(.002*39,-.003*39,.05*39)

def test_079_corrige_y_signo_invertido_se_descarta():
 f=train(.79);assert f.verificado and f.correccion_necesaria and f.usando_imu
 bad=train(-.79);assert bad.verificado is False and not bad.correccion_necesaria and not bad.usando_imu
 assert bad.pose==(.002*39,-.003*39,.05*39)

@pytest.mark.parametrize('slope',[.90,1.10])
def test_incluye_limite_de_010(slope):
 assert train(slope).correccion_necesaria

@pytest.mark.parametrize('slope',[1.02,-.79,None])
def test_tf_original_identico_byte_por_byte_tambien_antes_del_chequeo(slope):
 sent=[];fake=N(fusion=RumboImu(),odom='odom',base='base_footprint',informar=lambda:None,pub_tf=N(publish=lambda msg:sent.append(serialize_message(msg))))
 for k in range(60):
  t=.1*k
  if slope is not None:fake.fusion.agregar_imu(t,slope*.05*k)
  tr=TransformStamped();tr.header.stamp.sec=k//10;tr.header.stamp.nanosec=(k%10)*100000000
  tr.header.frame_id='odom';tr.child_frame_id='base_footprint'
  tr.transform.translation.x=math.sin(k/10);tr.transform.translation.y=math.cos(k/10);tr.transform.translation.z=.02
  tr.transform.rotation.z=math.sin(.05*k/2);tr.transform.rotation.w=math.cos(.05*k/2)
  # También conservar Z, roll/pitch originales y transformaciones ajenas.
  tr.transform.rotation.x=.0001
  other=TransformStamped();other.header.frame_id='base_footprint';other.child_frame_id='sensor';other.transform.rotation.w=1.
  msg=TFMessage(transforms=[tr,other]);before=serialize_message(msg)
  OdometriaImu.al_tf(fake,msg)
  assert sent[-1]==before and serialize_message(msg)==before
 assert not fake.fusion.correccion_necesaria

def test_reset_requiere_volver_a_medir():
 f=train(.79);f.agregar_imu(-1.,0.)
 assert f.verificado is None and not f.correccion_necesaria
