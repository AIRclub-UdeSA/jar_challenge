"""Rumbo del IMU sobre la traslación de las ruedas (sobregiro del robot real)."""
import math
import pytest
from equipo_jar.odometria_imu import RumboImu


def girar_y_avanzar(f, escala_ruedas=1.25, signo_imu=1., imu_hasta=1e9, dt=.1):
    """Gira pi/2 en el lugar y avanza 1 m; las ruedas sobregiran `escala_ruedas`."""
    t, real, xr, yr = 0., 0., 0., 0.
    for paso in range(60):
        t += dt
        if paso < 30:
            real += math.pi/2/30
        else:
            xr += math.cos(real*escala_ruedas)*1/30; yr += math.sin(real*escala_ruedas)*1/30
        if t <= imu_hasta:
            f.agregar_imu(t-.03, signo_imu*real+.7)
        pose = f.agregar_ruedas(t, xr, yr, real*escala_ruedas)
    return pose


def test_sin_imu_reproduce_las_ruedas():
    x, y, a = girar_y_avanzar(RumboImu(), imu_hasta=-1)
    assert a == pytest.approx(math.pi/2*1.25)
    assert (x, y) == pytest.approx((math.cos(a), math.sin(a)), abs=1e-6)


def test_imu_corrige_el_sobregiro_y_orienta_el_avance():
    f = RumboImu()
    x, y, a = girar_y_avanzar(f)
    assert f.verificado and f.pendiente == pytest.approx(.8, abs=.05)
    # El primer radián de giro se midió con ruedas: queda su sobregiro (~0,2 rad), no el total (0,39).
    assert a == pytest.approx(math.pi/2+.21, abs=.03)
    assert (x, y) == pytest.approx((math.cos(a), math.sin(a)), abs=.03)


def test_imu_invertido_se_descarta():
    f = RumboImu()
    x, y, a = girar_y_avanzar(f, signo_imu=-1.)
    assert f.verificado is False
    assert a == pytest.approx(math.pi/2*1.25)


def test_imu_quieto_mientras_las_ruedas_giran_se_descarta():
    f = RumboImu()
    t = 0.
    for k in range(20):
        t += .1
        f.agregar_imu(t, 0.)
        f.agregar_ruedas(t, 0., 0., .1*k)
    assert f.verificado is False


def test_imu_vencido_sigue_con_ruedas_sin_saltos_y_reancla():
    f = RumboImu(max_edad_imu_s=.5)
    t = 0.
    for k in range(20):  # verificación: giro con IMU 0,8 de las ruedas
        t += .1
        f.agregar_imu(t, .08*k); f.agregar_ruedas(t, 0., 0., .1*k)
    assert f.verificado
    rumbo = f.pose[2]
    for k in range(20, 30):  # IMU detenido: incrementos de ruedas
        t += .1
        f.agregar_ruedas(t, 0., 0., .1*k)
    assert not f.usando_imu and f.pose[2] == pytest.approx(rumbo+1., abs=.06)
    rumbo = f.pose[2]
    t += .1; f.agregar_imu(t, 5.); f.agregar_ruedas(t, 0., 0., 3.)
    t += .1; f.agregar_imu(t, 5.08); f.agregar_ruedas(t, 0., 0., 3.1)
    assert f.usando_imu and f.pose[2] == pytest.approx(rumbo+.1+.08, abs=.02)


def test_no_interpola_a_traves_de_un_hueco_del_imu():
    f = RumboImu(max_edad_imu_s=.5)
    f.agregar_imu(1., 0.); f.agregar_imu(2.5, 1.)
    assert f.rumbo_imu(2.) is None and f.rumbo_imu(2.5) == 1.


def test_rumbo_cruza_pi_sin_saltos():
    f = RumboImu()
    t, a = 0., 3.
    for k in range(40):
        t += .1; a += .05
        f.agregar_imu(t, math.remainder(a*.8, 2*math.pi))
        f.agregar_ruedas(t, 0., 0., math.remainder(a, 2*math.pi))
    assert f.verificado and f.pendiente == pytest.approx(.8, abs=.02)


def test_reinicio_del_reloj_reinicia_la_estimacion():
    f = RumboImu()
    girar_y_avanzar(f)
    f.agregar_imu(.05, 0.)
    assert f.agregar_ruedas(.1, 2., 3., .4) == (2., 3., .4)
    assert f.verificado is None
