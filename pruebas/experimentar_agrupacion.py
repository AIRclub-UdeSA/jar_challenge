"""Compare whole-position grouping on saved routes, without changing runtime."""
import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

from equipo_jar.cobertura_plan import (
    Camera, Pose, Settings, TravelCosts, _meets_order_checks, _order_checkpoints,
    reachable_cells, route_metrics, sampled_cells)
from evaluar_cobertura import read_grid
from experimentar_orden import dominates, summary
from metricas_cobertura import visibility_masks


def visits(choices):
    return sum(index == 0 or item[0] != choices[index - 1][0]
               for index, item in enumerate(choices))


def group_views(choices, start_cell, start_yaw, costs, settings, mode):
    current = list(choices)
    metrics = route_metrics(current, start_cell, start_yaw, costs, settings, 0)
    cells = tuple(dict.fromkeys(item[0] for item in choices))
    attempts = accepted = 0
    for cell in cells:
        indices = [i for i, item in enumerate(current) if item[0] == cell]
        if len(indices) < 2 or indices[-1] - indices[0] + 1 == len(indices):
            continue
        runs = []
        for index in indices:
            if not runs or index != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(index)
        selections = ([a + b for a, b in combinations(runs, 2)]
                      if mode.startswith('pairs-') else [indices])
        checks = _order_checkpoints(metrics.coverage_curve,
                                   settings.order_check_interval_s,
                                   settings.order_start_delay_s)
        best, best_metrics = current, metrics
        for selected in selections:
            selected_set = set(selected)
            block = [current[index] for index in selected]
            rest = [item for index, item in enumerate(current) if index not in selected_set]
            for ordered, index in ((ordered, index)
                                   for ordered in (block, list(reversed(block)))
                                   for index in range(len(rest) + 1)):
                # Do not split another position's already continuous visit.
                if 0 < index < len(rest) and rest[index - 1][0] == rest[index][0]:
                    continue
                attempts += 1
                trial = rest[:index] + ordered + rest[index:]
                candidate = route_metrics(trial, start_cell, start_yaw, costs, settings, 0)
                if candidate.duration >= metrics.duration - 1e-9:
                    continue
                if (candidate.duration + candidate.return_distance / settings.travel_speed
                        > metrics.duration + metrics.return_distance / settings.travel_speed + 1e-9):
                    continue
                if (mode != 'pairs-clock'
                        and candidate.mean_first_view_time > metrics.mean_first_view_time + 1e-9):
                    continue
                if mode.endswith('dominance'):
                    if not dominates(candidate.coverage_curve, metrics.coverage_curve):
                        continue
                elif not _meets_order_checks(candidate.coverage_curve, checks):
                    continue
                rank = ((candidate.mean_first_view_time, candidate.duration) if mode.endswith('mean')
                        else (candidate.duration, candidate.mean_first_view_time))
                best_rank = ((best_metrics.mean_first_view_time, best_metrics.duration)
                             if mode.endswith('mean') else
                             (best_metrics.duration, best_metrics.mean_first_view_time))
                if rank < best_rank:
                    best, best_metrics = trial, candidate
        if best is not current:
            current, metrics = best, best_metrics
            accepted += 1
    return current, dict(attempts=attempts, accepted=accepted,
                         visits_before=visits(choices), visits_after=visits(current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--simulator', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--modes', nargs='+', choices=(
        'time', 'mean', 'dominance', 'pairs-time', 'pairs-mean', 'pairs-dominance', 'pairs-clock'),
                        default=('time', 'mean', 'dominance'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'build'):
        parser.error('runtime output must stay under build/')
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.reference.read_text())
    settings, camera = Settings(**saved['settings']), Camera(**saved['camera'])
    records = []
    for case in saved['cases']:
        if args.cases and case['case'] not in args.cases:
            continue
        print(f"Preparing {case['case']}", flush=True)
        grid = read_grid(args.simulator / 'maps' / f"{case['map']}.yaml")
        start = Pose(**case['start'])
        poses = [Pose(**p) for p in case['poses']]
        free = {(i, j) for j in range(grid.height) for i in range(grid.width)
                if grid.free((i, j), settings.free_threshold)}
        dense = tuple(sorted(free, key=lambda c: (c[1], c[0])))
        targets = sampled_cells(free, settings.target_spacing, grid.resolution, settings.max_targets)
        dense_masks = visibility_masks(grid, poses, dense, settings, camera)
        masks = visibility_masks(grid, poses, targets, settings, camera)
        choices = [(grid.cell(p.x, p.y), p, mask) for p, mask in zip(poses, masks)]
        dense_by_pose = {p: mask for p, mask in zip(poses, dense_masks)}
        safe = grid.safe_cells(settings.robot_radius + settings.safety_margin, settings.free_threshold)
        reachable = reachable_cells(safe, grid.cell(start.x, start.y))
        costs = TravelCosts(reachable, [grid.cell(start.x, start.y)] + [c[0] for c in choices], grid.resolution)
        startup, budgets = case['intended']['startup_delay'], saved['budgets']
        all_dense = (1 << len(dense)) - 1
        dense_choices = [(cell, p, dense_by_pose[p]) for cell, p, _ in choices]
        row = dict(case=case['case'], reference=summary(
            grid, start, dense_choices, costs, settings, startup, budgets, all_dense), variants={})
        for mode in args.modes:
            before = time.perf_counter()
            result, counts = group_views(choices, grid.cell(start.x, start.y), start.yaw,
                                         costs, settings, mode)
            elapsed = time.perf_counter() - before
            assert sorted((p.x, p.y, p.yaw) for _, p, _ in result) == sorted(
                (p.x, p.y, p.yaw) for p in poses)
            assert counts['visits_after'] <= counts['visits_before']
            dense_result = [(cell, p, dense_by_pose[p]) for cell, p, _ in result]
            metrics = summary(grid, start, dense_result, costs, settings, startup, budgets, all_dense)
            metrics.update(counts, compute_seconds=elapsed,
                           indices=[poses.index(p) for _, p, _ in result])
            row['variants'][mode] = metrics
            print(f"{case['case']} {mode}: {counts['accepted']} groups; "
                  f"visits {counts['visits_before']}->{counts['visits_after']}; "
                  f"seconds {row['reference']['metrics']['duration']:.2f}->"
                  f"{metrics['metrics']['duration']:.2f}; CPU {elapsed:.3f}", flush=True)
        records.append(row)
        payload = dict(reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                       settings=asdict(settings), camera=asdict(camera), cases=records,
                       limits='Saved views and predicted costs; not navigation, detection or collision measurement.')
        (args.output / 'experiment.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
