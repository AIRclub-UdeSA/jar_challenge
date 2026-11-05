"""Reproducible offline comparison. Reads practice maps; never copies them to source."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

from PIL import Image
import yaml

from equipo_jar.cobertura_plan import (
    Camera, Grid, Pose, Settings, make_plan, reachable_cells, visibility_mask)


def read_grid(path):
    meta = yaml.safe_load(path.read_text())
    image = Image.open(path.parent / meta['image']).convert('L')
    width, height = image.size
    pixels = image.tobytes()
    data = []
    for j in range(height - 1, -1, -1):
        for pixel in pixels[j * width:(j + 1) * width]:
            value = pixel / 255 if meta.get('negate', 0) else 1 - pixel / 255
            data.append(100 if value > meta.get('occupied_thresh', 0.65)
                        else 0 if value < meta.get('free_thresh', 0.25) else -1)
    return Grid(width, height, float(meta['resolution']), data, Pose(*meta['origin']))


def camera_from_contract(simulator):
    contract = yaml.safe_load((simulator / 'config/real_robot_contract.yaml').read_text())
    mount = contract['simulator']['frames']['mounts']['cam_1_color_frame']['nominal']
    # Optical +Z is color-frame +X. Its horizontal heading is the mount's yaw.
    return Camera(mount['xyz'][0], mount['xyz'][1], mount['rpy'][2])


def full_grid_visibility(grid, plan, settings, camera):
    cells = [(i, j) for j in range(grid.height) for i in range(grid.width)
             if grid.free((i, j), settings.free_threshold)]
    centers = [grid.center(cell) for cell in cells]
    covered = 0
    for pose in plan.poses:
        covered |= visibility_mask(grid, grid.cell(pose.x, pose.y), pose.yaw,
                                   centers, settings, camera)
    return cells, covered


def plot_case(grid, plan, baseline, settings, dense_cells, dense_mask, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    transform = Affine2D().rotate(grid.origin.yaw).translate(grid.origin.x, grid.origin.y)
    extent = (0, grid.width * grid.resolution, 0, grid.height * grid.resolution)
    data = np.array(grid.data).reshape(grid.height, grid.width)
    corners = transform.transform([(0, 0), (extent[1], 0), (0, extent[3]), (extent[1], extent[3])])
    for ax in axes[:2]:
        ax.imshow(data, origin='lower', cmap='gray_r', vmin=0, vmax=100,
                  extent=extent, transform=transform + ax.transData)
        ax.set(xlim=(min(corners[:, 0]), max(corners[:, 0])),
               ylim=(min(corners[:, 1]), max(corners[:, 1])), aspect='equal',
               xlabel='map x [m]', ylabel='map y [m]')
    missed = [grid.center(cell) for index, cell in enumerate(dense_cells)
              if not dense_mask & (1 << index)]
    visible = [grid.center(cell) for index, cell in enumerate(dense_cells)
               if dense_mask & (1 << index)]
    for points, color, label, marker, size in ((visible, '#b2dfc1', 'Predicted visible', '.', 3),
                                             (missed, '#c52936', 'Not predicted visible', 'x', 10)):
        if points:
            axes[0].scatter(*zip(*points), c=color, label=label, marker=marker, s=size, linewidths=0.5)
    for cells, color, label in ((plan.diagnostics.outside_candidates, '#8e2ea8', 'Outside candidate model'),
                                (plan.diagnostics.deferred, '#f5a623', 'Visible but not selected')):
        if cells:
            points = [grid.center(cell) for cell in cells]
            axes[0].scatter(*zip(*points), facecolors='none', edgecolors=color, label=label, s=32)
    axes[0].set_title(f'Full-grid planar prediction: {dense_mask.bit_count()}/{len(dense_cells)}')
    axes[0].legend(loc='upper left', fontsize=7)
    for leg in plan.diagnostics.legs:
        points = [grid.center(cell) for cell in leg]
        axes[1].plot(*zip(*points), color='#5891c7', linewidth=0.8, alpha=0.5)
    labels = {}
    for index, pose in enumerate(plan.poses):
        axes[1].arrow(pose.x, pose.y, 0.13 * math.cos(pose.yaw), 0.13 * math.sin(pose.yaw),
                      width=0.007, head_width=0.05, length_includes_head=True, color='#087f8c')
        labels.setdefault((pose.x, pose.y), []).append(str(index + 1))
    for (x, y), indexes in labels.items():
        axes[1].text(x + 0.03, y + 0.05, '/'.join(indexes), fontsize=5, color='#9c2b10')
    axes[1].set_title(f'{len(plan.poses)} goals; grid estimate {plan.metrics.distance:.1f} m')
    for item, label, color in ((baseline, 'Initial rule', '#a1a1a1'), (plan, 'Improved', '#087f8c')):
        curve = item.metrics.coverage_curve
        axes[2].step([x for x, _ in curve], [100 * y / item.target_count for _, y in curve],
                     where='post', label=label, color=color)
    axes[2].set(title='Predicted search progress', xlabel='Estimated seconds', ylabel='Sampled cells [%]',
                ylim=(0, 102))
    axes[2].grid(alpha=0.2)
    axes[2].legend()
    fig.suptitle(destination.stem, fontsize=13)
    fig.text(0.5, 0.02, 'Planar visibility only. Grid paths are estimates; no Nav2 execution or detections measured.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def evaluate(simulator, output, settings):
    camera = camera_from_contract(simulator)
    records = []
    starts = (Pose(0.0, 0.0, 0.0), Pose(0.15, -0.1, 0.2), Pose(-0.1, 0.15, -0.2))
    names = ('laberinto_1', 'laberinto_simple', 'maze_1_6x5', 'maze_2_6x5',
             'maze_3_6x6', 'maze_4_metal_6x6')
    files = [simulator / 'maps' / (name + '.yaml') for name in names]
    output.mkdir(parents=True, exist_ok=True)
    for path in files:
        grid = read_grid(path)
        for index, start in enumerate(starts):
            case = f'{path.stem}_start_{index}'
            row = {'case': case, 'map': path.stem, 'start': asdict(start)}
            try:
                plans = {}
                variants = ('baseline', 'improved') if index % 2 == 0 else ('improved', 'baseline')
                for variant in variants:
                    before = time.perf_counter()
                    plans[variant] = make_plan(grid, start, settings, camera,
                                              refine=variant == 'improved', improve=variant == 'improved',
                                              diagnostics=True)
                    item = plans[variant]
                    row[variant] = {'compute_seconds': time.perf_counter() - before,
                                    'goals': len(item.poses), 'covered': item.predicted_count,
                                    'targets': item.target_count, **asdict(item.metrics)}
                item, baseline = plans['improved'], plans['baseline']
                row['unpolished'] = asdict(item.unpolished_metrics)
                row['outside_candidates'] = len(item.diagnostics.outside_candidates)
                row['deferred'] = len(item.diagnostics.deferred)
                row['added_positions'] = item.diagnostics.added_positions
                raw_free = {(i, j) for j in range(grid.height) for i in range(grid.width)
                            if grid.free((i, j), settings.free_threshold)}
                raw_component = reachable_cells(raw_free, grid.cell(start.x, start.y))
                row['outside_disconnected_from_start'] = sum(
                    cell not in raw_component for cell in item.diagnostics.outside_candidates)
                row['outside_cells'] = item.diagnostics.outside_candidates
                row['deferred_cells'] = item.diagnostics.deferred
                assert item.predicted_count >= baseline.predicted_count, 'sampled coverage regressed'
                assert set(baseline.poses) <= set(item.poses), 'an initial observation was removed'
                assert item.metrics.duration <= item.unpolished_metrics.duration + 1e-8
                assert item.metrics.mean_first_view_time <= item.unpolished_metrics.mean_first_view_time + 1e-8
                if index == 0:
                    dense_cells, dense_mask = full_grid_visibility(grid, item, settings, camera)
                    _, base_dense = full_grid_visibility(grid, baseline, settings, camera)
                    row['full_grid'] = {'targets': len(dense_cells), 'covered': dense_mask.bit_count(),
                                        'baseline_covered': base_dense.bit_count()}
                    assert dense_mask | base_dense == dense_mask, 'full-grid visibility regressed'
                    plot_case(grid, item, baseline, settings, dense_cells, dense_mask, output / f'{case}.png')
                print(f'{case}: sample {baseline.predicted_count}->{item.predicted_count}/{item.target_count}, '
                      f'meters {baseline.metrics.distance:.1f}->{item.metrics.distance:.1f}, '
                      f'seconds {baseline.metrics.duration:.1f}->{item.metrics.duration:.1f}, '
                      f'outside={row["outside_candidates"]}, deferred={row["deferred"]}', flush=True)
            except ValueError as exc:
                row['error'] = str(exc)
                print(f'{case}: rejected: {exc}', flush=True)
            records.append(row)
            (output / 'comparison.json').write_text(json.dumps({
                'platform': platform.platform(), 'processor': platform.processor(),
                'python': sys.version,
                'planner_sha256': hashlib.sha256(Path(sys.modules[make_plan.__module__].__file__).read_bytes()).hexdigest(),
                'settings': asdict(settings), 'camera': asdict(camera), 'cases': records,
                'limits': 'One run per variant and case; timings include diagnostic grid paths. '
                          'Same planar visibility model, not a detector or navigation benchmark.'}, indent=2) + '\n')
    good = [row for row in records if 'improved' in row and 'error' not in row]
    columns = ['case', 'coverage_before', 'coverage_after', 'distance_before_m', 'distance_after_m',
               'duration_before_s', 'duration_after_s', 'first_view_before_s', 'first_view_after_s']
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in good:
            a, b = row['baseline'], row['improved']
            writer.writerow([row['case'], a['covered'] / a['targets'], b['covered'] / b['targets'],
                             a['distance'], b['distance'], a['duration'], b['duration'],
                             a['mean_first_view_time'], b['mean_first_view_time']])
    lines = ['# Local coverage comparison', '',
             'Planar predictions and grid travel estimates; no executed navigation or detections.', '',
             '| Case | Samples before → after | Metres before → after | Estimated seconds before → after |',
             '| --- | --- | --- | --- |']
    for row in good:
        a, b = row['baseline'], row['improved']
        lines.append(f'| {row["case"]} | {a["covered"]} → {b["covered"]}/{b["targets"]} | '
                     f'{a["distance"]:.1f} → {b["distance"]:.1f} | {a["duration"]:.1f} → {b["duration"]:.1f} |')
    lines.extend(['', f'{len(good)}/{len(records)} starts accepted. Rejected starts remain visible in JSON.',
                  'One run per variant and case; computation timings include diagnostic path reconstruction.',
                  'Camera mount comes from the simulator contract. Range still needs detector calibration.', '',
                  '## Residuals at the default start', '',
                  '| Map | Outside sampled candidate model | Disconnected even without inflation | Full-grid visible |',
                  '| --- | --- | --- | --- |'])
    for row in good:
        if 'full_grid' in row:
            dense = row['full_grid']
            lines.append(f'| {row["map"]} | {row["outside_candidates"]} | '
                         f'{row["outside_disconnected_from_start"]} | {dense["covered"]}/{dense["targets"]} |')
    lines.extend(['', 'Time estimates omit acceleration, localization delays, new obstacles, '
                  'additional path-following rotations and perception workload. Return distance is listed '
                  'separately; mission owns the return decision and the real time budget.'])
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', required=True, type=Path, help='installed simulator share directory')
    parser.add_argument('--output', type=Path, default=Path('build/verificacion/cobertura/evaluacion'))
    parser.add_argument('--params', type=Path, default=Path('src/equipo_jar/params/params.yaml'))
    args = parser.parse_args()
    values = yaml.safe_load(args.params.read_text())['cobertura']['ros__parameters']
    settings = Settings(**{name: values.get(name, value) for name, value in asdict(Settings()).items()})
    evaluate(args.simulator, args.output, settings)


if __name__ == '__main__':
    main()
