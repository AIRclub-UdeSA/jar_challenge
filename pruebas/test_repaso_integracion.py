from equipo_jar.cobertura_plan import Camera, Settings
from equipo_jar.cobertura_repaso import pending_samples, propose_extra_views
from equipo_jar.historial_vistas import ObservedView
from equipo_jar.mapa import Pose
from test_repaso_cobertura import room, outside_history


def test_invalid_historical_base_cannot_credit_the_room_beyond_the_wall():
    static, live = room(occluder=False)
    camera = Camera(0.1, 0.0, 0.0)
    before = pending_samples(static, live, (), camera, Settings())
    for x in (3.025, 2.975, -0.025):
        after = pending_samples(static, live, (ObservedView(Pose(x, 0.525, 0.0), 10.0),),
                                camera, Settings())
        assert after == before


def test_remembered_occlusion_does_not_retroactively_credit_old_views():
    static, memory = room()
    _, live = room(occluder=False)
    observed = tuple(ObservedView(v.pose, float(i)) for i, v in enumerate(outside_history()))
    before = pending_samples(static, memory, observed, Camera(), Settings())
    after = pending_samples(static, live, observed, Camera(), Settings(), occlusion_grid=memory)
    assert after == before
    result = propose_extra_views(static, live, Pose(1, 2, 0), Pose(1, 2, 0), observed,
                                 (), Camera(), Settings(), limit=6, failed_shift_m=0.4,
                                 occlusion_grid=memory)
    assert result.pending_count == len(before)
    assert result.views
