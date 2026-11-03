"""Heading fix after turns: when the heading is in doubt and when it is trusted again."""
import math
import random

from equipo_jar.correccion_giro import CorreccionGiro

P = {'actualizaciones_giro': 3, 'giro_minimo_rad': 0.3, 'girando_rads': 0.3,
     'max_correccion_s': 3.0, 'max_duda_s': 15.0}


def girar(c, t, yaw, hasta, wz=1.0, dt=0.1):
    """Odometry of a turn in place from yaw to `hasta`; returns (t, yaw) at the end."""
    paso = math.copysign(abs(wz) * dt, hasta - yaw)
    while abs(hasta - yaw) > abs(paso):
        t += dt
        yaw += paso
        c.con_odom(t, yaw, math.copysign(abs(wz), paso))
    return t, yaw


def quieto(c, t, yaw, segundos, dt=0.1):
    for _ in range(round(segundos / dt)):
        t += dt
        c.con_odom(t, yaw, 0.0)
    return t


def manejar(c, t, yaw, segundos, wz_de, dt=0.1):
    """Drive with wz = wz_de(k) per sample; returns (t, yaw, seconds in doubt)."""
    en_duda = 0.0
    for k in range(round(segundos / dt)):
        wz = wz_de(k)
        t += dt
        yaw += wz * dt
        c.con_odom(t, yaw, wz)
        en_duda += dt if c.en_duda else 0.0
    return t, yaw, en_duda


def test_big_turn_is_in_doubt_until_updates_after_it():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 1.57)
    assert c.en_duda
    c.con_actualizacion_amcl(t - 0.5)        # scan taken during the turn: does not count
    t = quieto(c, t, yaw, 0.2)
    assert c.pedir_actualizacion()
    for k in range(3):
        assert c.en_duda
        c.con_actualizacion_amcl(t + 0.01 * k)
    assert not c.en_duda and not c.pedir_actualizacion()
    assert c.correcciones == 1 and c.vencidas == 0


def test_small_turns_do_not_add_up():
    c = CorreccionGiro(dict(P))
    t, yaw = 0.0, 0.0
    for _ in range(3):
        t, yaw = girar(c, t, yaw, yaw + 0.2)
        t = quieto(c, t, yaw, 0.3)
        assert not c.en_duda


def test_turning_again_restarts_the_count():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 1.0)
    t = quieto(c, t, yaw, 0.2)
    c.con_actualizacion_amcl(t)
    c.con_actualizacion_amcl(t)
    t, yaw = girar(c, t, yaw, 2.0)
    assert c.en_duda and c.actualizaciones == 0
    t = quieto(c, t, yaw, 0.2)
    for _ in range(3):
        c.con_actualizacion_amcl(t)
    assert not c.en_duda


def test_driving_on_after_the_turn_still_fixes_it():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 1.0)
    t += 0.1
    c.con_odom(t, yaw, 0.1)                  # drives on with a small heading correction
    assert c.en_duda and c.pedir_actualizacion()
    for _ in range(3):
        c.con_actualizacion_amcl(t + 0.05)
    assert not c.en_duda


def test_gives_up_after_max_time():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 1.0)
    t = quieto(c, t, yaw, 3.5)               # no AMCL updates arrive
    assert not c.en_duda and c.vencidas == 1


def test_wobble_while_driving_is_not_a_turn():
    c = CorreccionGiro(dict(P))
    azar = random.Random(1)
    _, _, en_duda = manejar(c, 0.0, 0.0, 120.0, lambda _k: azar.gauss(0.0, 0.1))
    assert en_duda == 0.0


def test_s_curve_is_not_a_turn():
    c = CorreccionGiro(dict(P))
    # Left then right at 0.4 rad/s, 0.5 s each: 0.2 rad per side, cancels out.
    _, _, en_duda = manejar(c, 0.0, 0.0, 30.0, lambda k: 0.4 if (k // 5) % 2 else -0.4)
    assert en_duda == 0.0


def test_wz_dithering_around_the_threshold_does_not_hold_the_doubt():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 1.0)
    t, yaw, en_duda = manejar(c, t, yaw, 30.0, lambda k: 0.35 if k % 2 else 0.25)
    assert en_duda <= P['max_correccion_s'] + 0.1   # no AMCL updates: timed out once
    assert c.vencidas == 1 and not c.en_duda


def test_doubt_has_a_total_cap_on_a_long_spin():
    c = CorreccionGiro(dict(P))
    t, yaw = girar(c, 0.0, 0.0, 20.0)        # 20 s spinning, never ends
    assert c.vencidas >= 1
    assert t - (c.inicio_duda or t) <= P['max_duda_s']


def test_off_never_doubts():
    c = CorreccionGiro(dict(P, actualizaciones_giro=0))
    girar(c, 0.0, 0.0, 3.0)
    assert not c.en_duda and not c.pedir_actualizacion()


def test_live_param_change_mid_fix():
    p = dict(P)
    c = CorreccionGiro(p)
    t, yaw = girar(c, 0.0, 0.0, 1.0)
    t = quieto(c, t, yaw, 0.2)
    c.con_actualizacion_amcl(t)
    p['actualizaciones_giro'] = 1            # lowered below the updates already done
    c.con_actualizacion_amcl(t)
    assert not c.en_duda and c.correcciones == 1
    t, yaw = girar(c, t, yaw, 2.0)
    assert c.en_duda
    p['actualizaciones_giro'] = 0            # turned off mid-fix
    t = quieto(c, t, yaw, 0.1)
    assert not c.en_duda and not c.pedir_actualizacion()


def test_angle_wraps_around_pi():
    c = CorreccionGiro(dict(P))
    c.con_odom(0.0, 3.1, 1.0)
    c.con_odom(0.1, -3.1, 1.0)               # 0.08 rad across +-pi, not 6.2
    assert abs(c.girado) < 0.1
