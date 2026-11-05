"""Audit explicit goal records; never infer success or measured poses from a plan."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import yaml

from equipo_jar.cobertura_plan import Camera, Settings, sampled_cells
from ejecucion_cobertura import audit_records, records_from_payload
from evaluar_cobertura import read_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--yaw-error-deg', type=float, default=10.0)
    parser.add_argument('--position-error-m', type=float, default=0.0)
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / 'build'):
        parser.error('output must stay under this checkout\'s build directory')
    if (not math.isfinite(args.yaw_error_deg) or not 0 <= args.yaw_error_deg < 180
            or not math.isfinite(args.position_error_m) or args.position_error_m < 0):
        parser.error('error samples must be finite and nonnegative; yaw must be below 180')
    payload = json.loads(args.records.read_text())
    records = records_from_payload(payload)
    camera = Camera(**payload['camera'])
    params = yaml.safe_load((root / 'src/equipo_jar/params/params.yaml').read_text())
    settings = Settings(**{key: params['cobertura']['ros__parameters'].get(key, value)
                           for key, value in asdict(Settings()).items()})
    grid = read_grid(args.map)
    cells = {(i, j) for j in range(grid.height) for i in range(grid.width)
             if grid.free((i, j), settings.free_threshold)}
    cells = (tuple(sorted(cells, key=lambda c: (c[1], c[0]))) if args.dense else
             tuple(sampled_cells(cells, settings.target_spacing, grid.resolution, settings.max_targets)))
    p, yaw = args.position_error_m, math.radians(args.yaw_error_deg)
    offsets = tuple(sorted({(dx, dy, dyaw) for dx, dy in ((0, 0), (-p, 0), (p, 0), (0, -p), (0, p))
                            for dyaw in (-yaw, 0, yaw)}))
    report = audit_records(grid, records, cells, settings, camera,
                           (60, 120, 300, 480, 600), offsets=offsets)
    for key in ('nominal_mask', 'sampled_robust_mask'):
        del report[key]
    report.update(source=payload['source'], pose_source=payload['pose_source'],
                   clock=payload['clock'], time_origin=payload['time_origin'],
                   target_cells=len(cells), target_mode='dense' if args.dense else 'sampled',
                   error_samples=offsets,
                   limits='Geometric visibility at successful goal poses, not processed images. '
                   'No transit/inspection views beyond the supplied records. Commanded/predicted '
                   'poses must not be called measured. Sampled errors do not guarantee all '
                   'intermediate errors. No collision or score inference.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['succeeded']} successful, {report['failed']} failed, {report['canceled']} canceled; "
          f"{report['nominal']}/{len(cells)} geometric cells visible", flush=True)


if __name__ == '__main__':
    main()
