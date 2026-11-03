"""Paired local timing comparison against a saved pure planner revision."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import yaml

from equipo_jar import cobertura_plan as current
from evaluar_cobertura import camera_from_contract, read_grid


def load_reference(path):
    spec = importlib.util.spec_from_file_location('coverage_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--simulator', required=True, type=Path)
    parser.add_argument('--params', type=Path, default=Path('src/equipo_jar/params/params.yaml'))
    parser.add_argument('--output', type=Path, default=Path('build/verificacion/cobertura/rendimiento'))
    parser.add_argument('--maps', nargs='+', default=('laberinto_1', 'laberinto_simple', 'maze_1_6x5'))
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--allow-reordering', action='store_true',
                        help='require identical view sets and stable output per version; allow a new order')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'build'):
        parser.error('runtime output must stay under this checkout\'s build directory')
    if args.repetitions < 2:
        parser.error('use at least two paired repetitions')
    reference = load_reference(args.reference)
    params = yaml.safe_load(args.params.read_text())['cobertura']['ros__parameters']
    values = {key: params.get(key, value) for key, value in asdict(current.Settings()).items()}
    camera_values = asdict(camera_from_contract(args.simulator))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.maps:
        source = read_grid(args.simulator / 'maps' / f'{name}.yaml')
        inputs = {}
        for label, module in (('reference', reference), ('current', current)):
            grid = module.Grid(source.width, source.height, source.resolution,
                               list(source.data), module.Pose(**asdict(source.origin)))
            inputs[label] = (module.make_plan, grid, module.Pose(0.0, 0.0, 0.0),
                             module.Settings(**{key: values[key] for key in asdict(module.Settings())}),
                             module.Camera(**camera_values))
        expected = None
        expected_by_version = {}
        row = dict(map=name, reference=[], current=[])
        for repetition in range(-1, args.repetitions):
            order = ('reference', 'current') if repetition % 2 == 0 else ('current', 'reference')
            for label in order:
                function, *arguments = inputs[label]
                before = time.perf_counter()
                result = function(*arguments)
                seconds = time.perf_counter() - before
                signature = ([(p.x, p.y, p.yaw) for p in result.poses], result.predicted_count,
                             result.target_count, asdict(result.metrics))
                shared = (sorted(signature[0]), *signature[1:3]) if args.allow_reordering else signature
                if expected is None:
                    expected = shared
                assert shared == expected, f'Planner output changed beyond allowed order: {name}/{label}'
                expected_by_version.setdefault(label, signature)
                assert signature == expected_by_version[label], f'Non-deterministic output: {name}/{label}'
                if repetition >= 0:
                    row[label].append(seconds)
                print(f'{name} {label} {"warmup" if repetition < 0 else repetition + 1}: '
                      f'{seconds:.3f} s; {"same views" if args.allow_reordering else "identical output"}', flush=True)
        row['reference_median'] = statistics.median(row['reference'])
        row['current_median'] = statistics.median(row['current'])
        row['median_speedup'] = row['reference_median'] / row['current_median']
        records.append(row)
    payload = dict(
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        current_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        settings=values, camera=camera_values, cases=records,
        route_comparison='same view set' if args.allow_reordering else 'identical route and metrics',
        limits='One warmup and alternating paired repetitions on this machine; '
               'not a statistical or hardware-independent runtime guarantee. No diagnostics.')
    (args.output / 'timing.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Paired planner runtime comparison', '', payload['limits'], '',
             '| Map | Reference median s | Current median s | Speedup |',
             '| --- | --- | --- | --- |']
    for row in records:
        lines.append(f'| {row["map"]} | {row["reference_median"]:.3f} | '
                     f'{row["current_median"]:.3f} | {row["median_speedup"]:.2f}x |')
    guarantee = ('Every run asserts identical view sets and target counts, with deterministic order/metrics per version.'
                 if args.allow_reordering else
                 'Every run asserts identical poses, predicted target counts and route metrics.')
    lines.extend(['', guarantee])
    (args.output / 'timing.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
