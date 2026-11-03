from test_historial_mision import mission, map_message
from test_mision_repaso import search, advance


def pending_plan(search):
    node, _, _, executor, _ = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()


def test_navigation_only_change_refreshes_routes_without_recomputing_views(search):
    node, _, _, executor, navigation = search
    calls = []
    original = node.extra.provider
    def provider(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    node.extra.provider = provider
    pending_plan(search)
    changed = map_message()
    changed.data[0] = 99
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    assert node.extra.views is not None
    assert node.extra.field is None
    executor.finish()
    node.step_logic()
    assert len(calls) == 1
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == ''


def test_newly_blocked_endpoint_is_not_dispatched_from_retained_views(search):
    node, _, _, executor, navigation = search
    pending_plan(search)
    changed = map_message()
    i, j = node.grid.cell(2.0, 2.4)
    changed.data[j * 20 + i] = 99
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert node.extra.views is not None
    executor.finish()
    node.step_logic()
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_disconnected_home_does_not_grant_a_free_return_estimate(search):
    node, _, _, executor, navigation = search
    pending_plan(search)
    changed = map_message()
    for i in range(20):
        changed.data[5 * 20 + i] = 99
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert node.extra.views is not None
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    advance(search, 6.0)
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_changed_optical_mask_requires_another_full_plan(search):
    node, _, _, executor, navigation = search
    calls = []
    original = node.extra.provider
    def provider(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    node.extra.provider = provider
    pending_plan(search)
    changed = map_message()
    changed.data[0] = 100
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert node.extra.views is None
    assert not navigation.sent
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(calls) == 2


def test_another_navigation_change_retries_only_the_cheap_route_refresh(search):
    node, _, _, executor, navigation = search
    calls = []
    original = node.extra.provider
    def provider(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    node.extra.provider = provider
    pending_plan(search)
    changed = map_message()
    changed.data[0] = 99
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    executor.finish()
    changed.data[1] = 99
    node.on_costmap(changed)
    node.step_logic()
    assert not navigation.sent
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(calls) == 1
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == ''


def test_disabling_search_during_refresh_never_dispatches_the_retained_views(search):
    node, _, _, executor, navigation = search
    pending_plan(search)
    changed = map_message()
    changed.data[0] = 99
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert node.extra.views is not None
    node.p['extra_search_enabled'] = False
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert node.extra.views is None
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)
