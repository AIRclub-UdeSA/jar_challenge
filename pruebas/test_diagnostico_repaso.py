from types import SimpleNamespace

import pytest
from std_msgs.msg import Bool

from test_historial_mision import mission, map_message
from test_mision_repaso import search, advance


@pytest.fixture
def diagnostic(search, monkeypatch):
    node = search[0]
    original = node.get_logger()
    messages = []
    def info(message, **kwargs):
        messages.append(message)
    logger = SimpleNamespace(info=info, warning=original.warning, warn=original.warn, error=original.error)
    monkeypatch.setattr(node, 'get_logger', lambda: logger)
    return search, messages


def enter(search):
    node, _, _, executor, _ = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()


def test_submit_and_completion_identify_job_and_computation_times(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, _ = search
    enter(search)
    executor.finish()
    node.step_logic()
    assert any('event=submit' in m and 'kind=plan' in m and 'job=' in m and 'generation=' in m for m in messages)
    assert any('event=result' in m and 'kind=plan' in m and 'route_ms=' in m and 'planner_ms=' in m and 'compute_ms=' in m and 'queue_ms=' in m for m in messages)
    assert any('event=accept' in m and 'kind=plan' in m for m in messages)


def test_changed_navigation_mask_records_discard_without_dispatch(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    enter(search)
    changed = map_message()
    changed.data[0] = 100
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    assert any('event=discard' in m and 'reason=navigation_changed' in m and 'changed_cells=1' in m for m in messages)


def test_localization_wait_is_distinct_from_worker_time_and_does_not_spam(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, _ = search
    node.step_logic()
    executor.finish()
    node.on_localization_ready(Bool(data=False))
    advance(search, 1.1)
    node.step_logic()
    node.step_logic()
    node.step_logic()
    waits = [m for m in messages if 'event=wait' in m and 'reason=localization_not_ready' in m]
    assert len(waits) == 1
    advance(search, 1.6)
    node.on_localization_ready(Bool(data=True))
    node.step_logic()
    assert executor.jobs
    assert any('event=submit' in m and 'kind=plan' in m and 'wait_s=1.600' in m for m in messages)


def test_timeout_reports_running_job_without_changing_return_policy(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    enter(search)
    executor.jobs[0][0].set_running_or_notify_cancel()
    advance(search, 6.0)
    node.step_logic()
    assert node.state == 'returning'
    assert any('event=timeout' in m and 'pending_kind=plan' in m and 'pending_status=running' in m for m in messages)
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)
    returning = node.current_goal
    executor.finish()
    node.step_logic()
    assert node.current_goal is returning
    assert node.extra.field is None and node.extra.views is None
    assert any('event=result' in m and 'kind=plan' in m and 'compute_ms=' in m for m in messages)
    assert any('event=discard' in m and 'reason=inactive' in m for m in messages)


def test_provider_error_records_stage_and_preserves_bounded_return(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, _ = search
    def fail(*args, **kwargs):
        raise ValueError('probe')
    node.extra.provider = fail
    enter(search)
    executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert any('event=result' in m and 'stage=planner' in m and 'outcome=ValueError' in m and 'compute_ms=' in m for m in messages)


def test_disabled_feature_produces_no_search_diagnostics(mission, monkeypatch):
    node, _, _ = mission
    messages = []
    original = node.get_logger()
    monkeypatch.setattr(node, 'get_logger', lambda: SimpleNamespace(
        info=lambda message, **kwargs: messages.append(message),
        warning=original.warning, warn=original.warn, error=original.error))
    node.step_logic()
    assert not any('Extra search event=' in m for m in messages)
    assert node.extra.executor is None


def test_computation_metrics_measure_wall_time_separately_from_simulated_wait(diagnostic, monkeypatch):
    from equipo_jar import busqueda_extra
    search, messages = diagnostic
    node, _, _, executor, _ = search
    wall = SimpleNamespace(seconds=0.0)
    build = busqueda_extra.build_route_field
    provider = node.extra.provider
    def route(*args, **kwargs):
        result = build(*args, **kwargs)
        wall.seconds += 0.25
        return result
    def plan(*args, **kwargs):
        result = provider(*args, **kwargs)
        wall.seconds += 0.75
        return result
    monkeypatch.setattr(busqueda_extra, 'perf_counter', lambda: wall.seconds)
    monkeypatch.setattr(busqueda_extra, 'build_route_field', route)
    node.extra.provider = plan
    enter(search)
    wall.seconds += 2.0
    executor.finish()
    node.step_logic()
    result = next(m for m in messages if 'event=result' in m and 'kind=plan' in m)
    assert 'route_ms=250.000' in result
    assert 'planner_ms=750.000' in result
    assert 'compute_ms=1000.000' in result
    assert 'queue_ms=2000.000' in result


def test_optical_change_is_distinct_from_navigation_change(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    changed = map_message()
    changed.data[0] = 99
    node.on_costmap(changed)
    node.step_logic()
    changed.data[0] = 100
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    assert any('event=discard' in m and 'kind=plan' in m and 'reason=optical_changed' in m for m in messages)


def test_map_replacement_records_obsolete_generation(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    enter(search)
    changed = map_message()
    changed.data[0] = 100
    node.al_recibir_mapa(changed)
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    assert any('event=discard' in m and 'reason=generation_changed' in m for m in messages)


def test_localization_delay_and_physical_mask_changes_explain_timeout(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    node.on_localization_ready(Bool(data=False))
    advance(search, 1.1)
    node.step_logic()
    advance(search, 1.6)
    node.on_localization_ready(Bool(data=True))
    node.step_logic()
    changed = map_message()
    changed.data[0] = 100
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    changed.data[1] = 100
    node.on_costmap(changed)
    node.step_logic()
    advance(search, 2.2)
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)
    assert any('event=wait' in m and 'reason=localization_not_ready' in m for m in messages)
    assert sum('event=discard' in m and 'kind=plan' in m and 'reason=navigation_changed' in m for m in messages) == 2
    assert any('event=timeout' in m and 'last_discard=navigation_changed' in m for m in messages)



def test_disabled_search_drains_the_old_job_without_accepting_it(diagnostic):
    search, messages = diagnostic
    node, _, _, executor, navigation = search
    enter(search)
    executor.jobs[0][0].set_running_or_notify_cancel()
    node.p['extra_search_enabled'] = False
    node.step_logic()
    returning = node.current_goal
    executor.finish()
    node.step_logic()
    assert node.current_goal is returning
    assert node.extra.field is None and node.extra.views is None
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)
    assert any('event=result' in m and 'kind=plan' in m for m in messages)
    assert any('event=discard' in m and 'reason=inactive' in m for m in messages)
