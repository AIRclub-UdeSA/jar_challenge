"""Compare fixed-view ordering moves against saved coverage plans.

No shipped planner changes. All views are retained and visibility uses every
free cell center. Results are geometric predictions, not successful detection.
"""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import math
import time

from equipo_jar.cobertura_plan import (
    Pose, Settings, TravelCosts, reachable_cells, route_metrics, sampled_cells,
    visibility_mask)
from evaluar_cobertura import read_grid
from metricas_cobertura import at_budget, route_trace, visibility_masks


def dominates(curve, reference):
    """At every reference event, the new curve has seen at least as much."""
    index = 0
    for seconds, required in reference:
        while index + 1 < len(curve) and curve[index + 1][0] <= seconds + 1e-9:
            index += 1
        if curve[index][1] < required:
            return False
    return True


def value_at(curve, seconds):
    index = 0
    while index + 1 < len(curve) and curve[index + 1][0] <= seconds + 1e-9:
        index += 1
    return curve[index][1]


def relocate(choices, start, costs, settings, sampled, mode, guard_times=()):
    """One sweep; retain every view and reject total-time/sample-mean regressions."""
    current = list(choices)
    metrics = route_metrics(current, start[0], start[1], costs, settings, 0)
    sample_metrics = route_metrics(sampled, start[0], start[1], costs, settings, 0)
    sample_mask = {entry[1]: entry[2] for entry in sampled}
    attempts = accepted = 0
    for original in tuple(choices):
        first = current.index(original)
        best, best_metrics, best_sample = current, metrics, sample_metrics
        rest = current[:first] + current[first + 1:]
        for last in range(len(current)):
            if last == first:
                continue
            attempts += 1
            trial = rest[:last] + [original] + rest[last:]
            candidate = route_metrics(trial, start[0], start[1], costs, settings, 0)
            if candidate.duration > metrics.duration + 1e-9:
                continue
            if (candidate.duration + candidate.return_distance / settings.travel_speed
                    > metrics.duration + metrics.return_distance / settings.travel_speed + 1e-9):
                continue
            earlier = candidate.mean_first_view_time < best_metrics.mean_first_view_time - 1e-9
            same_mean = abs(candidate.mean_first_view_time - best_metrics.mean_first_view_time) <= 1e-9
            if not (earlier or (same_mean and candidate.duration < best_metrics.duration - 1e-9)):
                continue
            if mode == 'dominance' and not dominates(candidate.coverage_curve, metrics.coverage_curve):
                continue
            if mode == 'budget' and any(value_at(candidate.coverage_curve, seconds)
                                       < value_at(metrics.coverage_curve, seconds)
                                       for seconds in guard_times):
                continue
            sample_trial = [(cell, pose, sample_mask[pose]) for cell, pose, _ in trial]
            candidate_sample = route_metrics(sample_trial, start[0], start[1], costs, settings, 0)
            if candidate_sample.mean_first_view_time > sample_metrics.mean_first_view_time + 1e-9:
                continue
            best, best_metrics, best_sample = trial, candidate, candidate_sample
        if best is not current:
            accepted += 1
            current, metrics, sample_metrics = best, best_metrics, best_sample
    return current, dict(attempts=attempts, accepted=accepted)


def summary(grid, start, choices, costs, settings, startup, budgets, target_mask):
    metrics = route_metrics(choices, grid.cell(start.x, start.y), start.yaw, costs, settings, 0)
    poses = [entry[1] for entry in choices]
    masks = [entry[2] for entry in choices]
    trace = route_trace(grid, start, poses, masks, range(len(poses)), costs, settings, startup)
    return dict(metrics=asdict(metrics), budgets={str(seconds): at_budget(
        trace, seconds, target_mask=target_mask, travel_speed=settings.travel_speed)
        for seconds in budgets})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--simulator', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('build/verificacion/cobertura/orden-experimentos'))
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--modes', nargs='+', choices=(
        'mean', 'dominance', 'budget', 'sample-mean', 'sample-dominance',
        'sample-budget', 'sample-grid30', 'sample-grid60', 'sample-phase60'),
        default=('mean', 'dominance', 'budget', 'sample-mean', 'sample-dominance', 'sample-budget'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'build'):
        parser.error('runtime output must stay under build/')
    saved = json.loads(args.reference.read_text())
    settings = Settings(**saved['settings'])
    from equipo_jar.cobertura_plan import Camera
    camera = Camera(**saved['camera'])
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for case in saved['cases']:
        if args.cases and case['case'] not in args.cases:
            continue
        print(f"Preparing {case['case']}", flush=True)
        grid = read_grid(args.simulator / 'maps' / f"{case['map']}.yaml")
        start = Pose(**case['start'])
        poses = [Pose(**pose) for pose in case['poses']]
        free = {(i, j) for j in range(grid.height) for i in range(grid.width)
                if grid.free((i, j), settings.free_threshold)}
        dense = tuple(sorted(free, key=lambda c: (c[1], c[0])))
        targets = sampled_cells(free, settings.target_spacing, grid.resolution, settings.max_targets)
        dense_masks = visibility_masks(grid, poses, dense, settings, camera)
        sample_masks = visibility_masks(grid, poses, targets, settings, camera)
        choices = [(grid.cell(p.x, p.y), p, mask) for p, mask in zip(poses, dense_masks)]
        sampled = [(grid.cell(p.x, p.y), p, mask) for p, mask in zip(poses, sample_masks)]
        safe = grid.safe_cells(settings.robot_radius + settings.safety_margin, settings.free_threshold)
        reachable = reachable_cells(safe, grid.cell(start.x, start.y))
        costs = TravelCosts(reachable, [grid.cell(start.x, start.y)] + [c[0] for c in choices], grid.resolution)
        startup = case['intended']['startup_delay']
        budgets = saved['budgets']
        target_mask = (1 << len(dense)) - 1
        original = summary(grid, start, choices, costs, settings, startup, budgets, target_mask)
        row = dict(case=case['case'], reference=original, variants={})
        for mode in args.modes:
            guard_times = tuple(max(0.0, b - startup) for b in budgets)
            objective = mode.removeprefix('sample-')
            if objective.startswith('grid'):
                step = float(objective.removeprefix('grid'))
                guard_times = tuple(step * k for k in range(
                    1, math.ceil(original['metrics']['duration'] / step) + 1))
                objective = 'budget'
            elif objective == 'phase60':
                step = 60.0
                guard_times = tuple(max(0.0, step * k - startup) for k in range(
                    1, math.ceil((original['metrics']['duration'] + startup) / step) + 1))
                objective = 'budget'
            before = time.perf_counter()
            reordered, counts = relocate(
                sampled if mode.startswith('sample-') else choices,
                (grid.cell(start.x, start.y), start.yaw), costs, settings, sampled,
                objective, guard_times)
            seconds = time.perf_counter() - before
            assert set(entry[1] for entry in reordered) == set(poses)
            dense_by_pose = {entry[1]: entry[2] for entry in choices}
            reordered = [(cell, pose, dense_by_pose[pose]) for cell, pose, _ in reordered]
            result = summary(grid, start, reordered, costs, settings, startup, budgets, target_mask)
            result.update(counts, compute_seconds=seconds,
                          indices=[poses.index(entry[1]) for entry in reordered])
            row['variants'][mode] = result
            print(f"{case['case']} {mode}: {counts['accepted']} moves; "
                  f"duration {original['metrics']['duration']:.2f}->{result['metrics']['duration']:.2f}; "
                  f"mean {original['metrics']['mean_first_view_time']:.2f}->"
                  f"{result['metrics']['mean_first_view_time']:.2f}; {seconds:.3f} s", flush=True)
        rows.append(row)
        payload = dict(reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       settings=asdict(settings), camera=asdict(camera), cases=rows,
                       limits='Fixed saved routes, successful views assumed, no actual navigation/detection. '
                              'Budget fractions use all free cell centers; timings are isolated relocation only.')
        (args.output / 'experiment.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
