"""Measure predicted visibility within time budgets and mission selection.

Only practice-map evaluation: no movement, reports or victim coordinates.
Runtime output stays under ignored build/. See the report's model limits.
"""
import argparse
import cProfile
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import pstats
import sys
import time

import yaml

from equipo_jar.cobertura_plan import (
    Pose, Settings, TravelCosts, make_plan, observation_cells, reachable_cells)
from evaluar_cobertura import camera_from_contract, read_grid
from ejecucion_cobertura import predict_goals, prediction_trace
from metricas_cobertura import (
    classify_outside, mask_for_cells, mission_indices, route_trace,
    summarize_trace, visibility_masks)


MAPS = ('laberinto_1', 'laberinto_simple', 'maze_1_6x5', 'maze_2_6x5',
        'maze_3_6x6', 'maze_4_metal_6x6')
STARTS = (Pose(0.0, 0.0, 0.0), Pose(0.15, -0.1, 0.2), Pose(-0.1, 0.15, -0.2))
LIMITS = (
    'Planar visibility prediction, not detector or navigation performance. '
    'All predicted mission goals are assumed successful; inspections, candidate boxes, '
    'cancellations and in-transit observations are omitted. The intended/mission '
    'fields retain the historical planner dwell/identity-only proxy for comparison. '
    'mission_execution uses the real command-position substitution and current settle_s; '
    'it credits geometric visibility at arrival, not processed images. Startup assumes TF is immediately available '
    'and planning overlaps the mission startup wait. The wall-CPU planning sample '
    'is a hypothetical 1x startup delay, not measured simulated-clock or robot latency. '
    'Return cost starts at the '
    'last completed view, not an interpolated position during a later trip. '
    'One runtime sample per case; diagnostic/oracle times are separate.')


def union_masks(masks, indices):
    mask = 0
    for index in indices:
        mask |= masks[index]
    return mask


def evaluate_case(grid, start, settings, camera, mission, budgets, diagnose):
    before = time.perf_counter()
    plan = make_plan(grid, start, settings, camera)
    planning_seconds = time.perf_counter() - before
    before = time.perf_counter()
    details = make_plan(grid, start, settings, camera, diagnostics=True)
    diagnostic_seconds = time.perf_counter() - before
    assert details.poses == plan.poses, 'diagnostics changed the generated route'
    free = {(i, j) for j in range(grid.height) for i in range(grid.width)
            if grid.free((i, j), settings.free_threshold)}
    cells = tuple(sorted(free, key=lambda c: (c[1], c[0])))
    start_cell = grid.cell(start.x, start.y)
    raw = reachable_cells(free, start_cell)
    safe = grid.safe_cells(settings.robot_radius + settings.safety_margin,
                           settings.free_threshold)
    reachable = reachable_cells(safe, start_cell)
    observation = observation_cells(grid, reachable, settings)
    before = time.perf_counter()
    masks = visibility_masks(grid, plan.poses, cells, settings, camera)
    dense_seconds = time.perf_counter() - before
    target_masks = dict(
        dense_all=(1 << len(cells)) - 1,
        dense_raw_connected=mask_for_cells(cells, raw),
        sampled_all=mask_for_cells(cells, set(details.diagnostics.targets)))
    indices = tuple(range(len(plan.poses)))
    selected = mission_indices(plan.poses, mission['same_point_distance_m'],
                               mission['same_point_angle_rad'])
    selected_set = set(selected)
    skipped = [index for index in indices if index not in selected_set]
    costs = TravelCosts(reachable, [start_cell] + [grid.cell(p.x, p.y) for p in plan.poses],
                        grid.resolution)
    startup = max(mission['start_wait_s'], planning_seconds)
    intended_trace = route_trace(grid, start, plan.poses, masks, indices, costs, settings, startup)
    mission_trace = route_trace(grid, start, plan.poses, masks, selected, costs, settings, startup)
    execution = predict_goals(grid, plan.poses, start, costs, settings, mission,
                              planning_s=planning_seconds)
    execution_trace = prediction_trace(grid, start, execution, cells, settings, camera, costs)
    intended_mask = union_masks(masks, indices)
    mission_mask = union_masks(masks, selected)
    execution_mask = execution_trace.events[-1].mask if execution_trace.events else 0
    assert (intended_mask & target_masks['sampled_all']).bit_count() == plan.predicted_count
    assert abs(intended_trace.duration - startup - plan.metrics.duration) < 1e-6
    result = dict(
        status='accepted', start=asdict(start),
        timings=dict(planning=planning_seconds, diagnostic_planning=diagnostic_seconds,
                     dense_visibility=dense_seconds),
        counts=dict(free=len(free), raw_connected=len(raw), safe=len(safe),
                    reachable_safe=len(reachable), targets=plan.target_count,
                    reachable_observation=len(observation),
                    candidate_positions=len(details.diagnostics.positions),
                    sampled_outside=len(details.diagnostics.outside_candidates),
                    sampled_deferred=len(details.diagnostics.deferred)),
        intended=summarize_trace(intended_trace, budgets, target_masks, settings),
        mission=summarize_trace(mission_trace, budgets, target_masks, settings),
        mission_execution=summarize_trace(execution_trace, budgets, target_masks, settings),
        execution_shifted=list(execution.shifted),
        execution_lost_dense=(intended_mask & ~execution_mask).bit_count(),
        execution_added_dense=(execution_mask & ~intended_mask).bit_count(),
        mission_indices=selected, mission_skipped=skipped,
        mission_lost_dense=(intended_mask & ~mission_mask).bit_count(),
        mission_lost_sampled=(intended_mask & ~mission_mask & target_masks['sampled_all']).bit_count(),
        poses=[asdict(p) for p in plan.poses])
    if diagnose:
        before = time.perf_counter()
        groups, witnesses = classify_outside(
            grid, details.diagnostics.outside_candidates, raw, observation, settings, camera)
        result['timings']['outside_oracle'] = time.perf_counter() - before
        result['outside'] = dict(counts={key: len(value) for key, value in groups.items()},
                                 cells=groups, witnesses=[asdict(p) for p in witnesses])
        # A witness in another safe component diagnoses the reachability limit;
        # it is not a usable goal from this start.
        before = time.perf_counter()
        other_observation = observation_cells(grid, safe - reachable, settings)
        other, other_witnesses = classify_outside(
            grid, groups['no_witness'], raw, other_observation, settings, camera)
        assert not other['raw_disconnected']
        result['timings']['other_safe_component_oracle'] = time.perf_counter() - before
        result['outside']['model_limits'] = dict(
            other_safe_component=len(other['candidate_sampling_gap']),
            no_safe_center_witness=len(other['no_witness']),
            other_component_cells=other['candidate_sampling_gap'],
            no_safe_center_cells=other['no_witness'],
            other_component_witnesses=[asdict(p) for p in other_witnesses])
    return result


def percent(value):
    return '-' if value is None else f'{100 * value:.1f}%'


def save_report(output, payload):
    (output / 'budget.json').write_text(json.dumps(payload, indent=2) + '\n')
    cases = [case for case in payload['cases'] if case['status'] == 'accepted']
    with (output / 'budget.csv').open('w', newline='') as stream:
        fields = ['case', 'route', 'target_set', 'seconds', 'completed', 'visible',
                  'targets', 'fraction', 'mean_fraction', 'return_from_last_completed_s']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for route in ('intended', 'mission', 'mission_execution'):
                for snapshot in case[route]['budgets'].values():
                    for target_set, values in snapshot.items():
                        writer.writerow(dict(case=case['case'], route=route,
                                             target_set=target_set, **values))
    lines = ['# Coverage budget evaluation', '', LIMITS, '',
             'Percentages use every free grid-cell center (dense), not just target samples.',
             'The connected column excludes raw-map components disconnected from the start.', '',
             '| Case | Commanded/plan goals | Dense @300 | Connected @300 | Dense @480 | Lost final cells | Command-time proxy s |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for case in cases:
        mission = case['mission_execution']
        budgets = mission['budgets']
        early = budgets.get('300.0', {})
        late = budgets.get('480.0', {})
        fraction = lambda snap, key: percent(snap.get(key, {}).get('fraction'))
        lines.append(
            f'| {case["case"]} | {mission["goals"]}/{case["intended"]["goals"]} | '
            f'{fraction(early, "dense_all")} | {fraction(early, "dense_raw_connected")} | '
            f'{fraction(late, "dense_all")} | {case["execution_lost_dense"]} | '
            f'{mission["duration"]:.1f} |')
    lines.extend(['', '## Missing sampled targets: exhaustive witness search', '',
                  'The model uses eligible observation centers, the current camera/range and heading grid.',
                  'No witness under this model is not a claim of physical impossibility.', '',
                  '| Case | Raw disconnected | Candidate sampling gap | No reachable witness | Other safe component | No safe-center witness |',
                  '| --- | --- | --- | --- | --- | --- |'])
    for case in cases:
        if 'outside' in case:
            groups = case['outside']['counts']
            limits = case['outside'].get('model_limits', {})
            lines.append(f'| {case["case"]} | {groups["raw_disconnected"]} | '
                         f'{groups["candidate_sampling_gap"]} | {groups["no_witness"]} | '
                         f'{limits.get("other_safe_component", "-")} | '
                         f'{limits.get("no_safe_center_witness", "-")} |')
    rejected = [case for case in payload['cases'] if case['status'] != 'accepted']
    lines.extend(['', f'{len(cases)} accepted; {len(rejected)} rejected. Rejections remain in JSON.'])
    for case in rejected:
        lines.append(f'- {case["case"]}: {case["error"]}')
    (output / 'budget.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('build/verificacion/cobertura/presupuesto'))
    parser.add_argument('--params', type=Path, default=Path('src/equipo_jar/params/params.yaml'))
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=MAPS)
    parser.add_argument('--starts', choices=('nominal', 'all'), default='all')
    parser.add_argument('--budgets', nargs='+', type=float, default=(60.0, 120.0, 300.0, 480.0))
    parser.add_argument('--diagnose', choices=('none', 'nominal', 'all'), default='nominal')
    parser.add_argument('--profile', action='store_true', help='profile one extra nominal planner run')
    args = parser.parse_args()
    if any(not 0 < value < float('inf') for value in args.budgets):
        parser.error('budgets must be finite and positive')
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'build'):
        parser.error('runtime output must stay under this checkout\'s build directory')
    args.output.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(args.params.read_text())
    settings = Settings(**{key: params['cobertura']['ros__parameters'].get(key, value)
                           for key, value in asdict(Settings()).items()})
    mission = params['mision']['ros__parameters']
    camera = camera_from_contract(args.simulator)
    payload = dict(platform=platform.platform(), python=sys.version, settings=asdict(settings),
                   camera=asdict(camera), mission_parameters=mission, budgets=args.budgets,
                   limits=LIMITS, cases=[], hashes={})
    for path in (root / 'src/equipo_jar/equipo_jar/cobertura_plan.py',
                 root / 'src/equipo_jar/equipo_jar/exploracion.py', Path(__file__),
                 root / 'src/equipo_jar/equipo_jar/mision.py',
                 Path(__file__).with_name('ejecucion_cobertura.py'),
                 Path(__file__).with_name('metricas_cobertura.py')):
        payload['hashes'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    starts = STARTS if args.starts == 'all' else STARTS[:1]
    for name in args.maps:
        grid = read_grid(args.simulator / 'maps' / f'{name}.yaml')
        for index, start in enumerate(starts):
            label = f'{name}_start_{index}'
            print(f'Evaluating {label}', flush=True)
            try:
                row = evaluate_case(grid, start, settings, camera, mission, args.budgets,
                                    args.diagnose == 'all' or (args.diagnose == 'nominal' and index == 0))
            except ValueError as exc:
                row = dict(status='rejected', error=str(exc), start=asdict(start))
            row.update(case=label, map=name)
            payload['cases'].append(row)
            save_report(args.output, payload)
            if row['status'] == 'accepted':
                print(f'{label}: commanded proxy {row["mission_execution"]["goals"]}/{row["intended"]["goals"]} goals, '
                      f'shifted {len(row["execution_shifted"])}, lost dense cells {row["execution_lost_dense"]}; '
                      f'outside {row.get("outside", {}).get("counts", "not diagnosed")}', flush=True)
            else:
                print(f'{label}: {row["error"]}', flush=True)
    if args.profile:
        grid = read_grid(args.simulator / 'maps' / f'{args.maps[0]}.yaml')
        profiler = cProfile.Profile()
        profiler.runcall(make_plan, grid, STARTS[0], settings, camera)
        profiler.dump_stats(str(args.output / 'planner.prof'))
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(30)
        (args.output / 'profile.txt').write_text(stream.getvalue())
    print(f'Report: {args.output / "budget.md"}', flush=True)
    return int(any(case['status'] != 'accepted' for case in payload['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
