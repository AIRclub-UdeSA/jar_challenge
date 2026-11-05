#!/usr/bin/env python3
"""Read a run log (logs_jar/log.txt, written by registro.py) quickly between runs.

Plain Python 3, no ROS and no extra packages: it runs on a Mac as is.

    python3 pruebas/ver_log.py log.txt                  # summary of the last run
    python3 pruebas/ver_log.py log.txt --run 1          # another run (--list shows them)
    python3 pruebas/ver_log.py log.txt --svg run.svg    # drawing: path, reports, stops
    python3 pruebas/ver_log.py log.txt --diff 1 2       # params that changed between two runs
    python3 pruebas/ver_log.py log.txt --grep Regreso   # every line that contains a text

What the summary answers (only params can change between runs, so it points at params):
- time line: /map, mission states, reports, /done, return to the start
- where the robot stood still for a long time (the judge ends the run after 60 s still)
- localization doubt: AMCL's standard deviation, and its worst moments
- WARN / ERROR lines per node, and the most common ones
"""
import argparse
import math
import re
import sys
from collections import Counter

# registro.py writes '<HH:MM:SS> <t since /map, 7 chars or "   -   "> <text>'
LINE = re.compile(r'^(\d\d:\d\d:\d\d) (.{7}) (.*)$')
LOGLINE = re.compile(r'^(DEBUG|INFO|WARN|ERROR|FATAL)\s*\[([^\]]+)\] (.*?)( \[\+(\d+) same\])?$')
POSE = re.compile(r'^POSE (\S+) (\S+) (\S+) sd_xy (\S+) (\S+)')
REPORT = re.compile(r'^REPORT (\S+) (\S+) frame (\S*)')
MAP = re.compile(r'^MAP (\d+)x(\d+) res (\S+) origin \((\S+), (\S+)\)')
START_POSE = re.compile(r'Pose de inicio (fija|\(provisoria\)): x=(\S+) y=(\S+)')
STATE = re.compile(r'Estado: <([^>]*)> -> <([^>]*)>')

STILL_WARN_S = 20.0        # a stop at least this long is listed (the judge's limit is 60 s)
STILL_MOVE_M = 0.03        # less than this between pose lines counts as standing still
DOUBT_SD_M = 0.15          # AMCL standard deviation above this is listed as doubtful


def to_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.nan
    return value


class Run:
    def __init__(self, header):
        self.header = header
        self.params = {}            # file -> list of lines
        self.lines = []             # (wall, t or None, text)
        self.events = []            # (t, kind, text) for the time line
        self.poses = []             # (t, x, y, yaw, sd)
        self.reports = []           # (t, x, y, frame)
        self.done = []
        self.map = None             # (w, h, res, ox, oy)
        self.start = None           # (x, y) from mission's 'Pose de inicio' line
        self.logs = []              # (t, level, node, text, repeated)
        self.full = False
        self.ended = False

    @property
    def last_t(self):
        times = [t for _, t, _ in self.lines if t is not None]
        return max(times) if times else None


def parse(lines):
    """Split the file into runs ('===== run start' lines)."""
    runs = []
    run = None
    params_file = None
    for raw in lines:
        raw = raw.rstrip('\n')
        m = LINE.match(raw)
        if not m:
            continue
        wall, t_text, text = m.groups()
        t = to_float(t_text) if t_text.strip() not in ('-', '') else None
        if t is not None and math.isnan(t):
            t = None
        if text.startswith('===== run start'):
            run = Run(text.strip('= ').replace('run start ', ''))
            runs.append(run)
            params_file = None
            continue
        if run is None:                       # lines before the first header
            run = Run('(no header)')
            runs.append(run)
        if text.startswith('----- params/'):
            params_file = text.strip('- ').replace('params/', '')
            run.params[params_file] = []
            continue
        if text.startswith('----- end of params'):
            params_file = None
            continue
        if params_file is not None:
            run.params[params_file].append(text)
            continue
        run.lines.append((wall, t, text))
        if text.startswith('===== log full'):
            run.full = True
        elif text.startswith('===== run end'):
            run.ended = True
        elif (pm := POSE.match(text)):
            x, y, yaw, sx, sy = map(to_float, pm.groups())
            run.poses.append((t, x, y, yaw, max(sx, sy)))
        elif (rm := REPORT.match(text)):
            run.reports.append((t, to_float(rm.group(1)), to_float(rm.group(2)), rm.group(3)))
            run.events.append((t, 'REPORT', f'x={rm.group(1)} y={rm.group(2)} frame={rm.group(3)}'))
        elif text.startswith('DONE'):
            run.done.append(t)
            run.events.append((t, 'DONE', ''))
        elif (mm := MAP.match(text)):
            w, h = int(mm.group(1)), int(mm.group(2))
            run.map = (w, h, *map(to_float, mm.group(3, 4, 5)))
            run.events.append((t, 'MAP', text[4:]))
        elif (lm := LOGLINE.match(text)):
            level, node, msg, _, rep = lm.groups()
            run.logs.append((t, level, node, msg, int(rep or 0)))
            if node == 'mision':
                if (sm := STATE.search(msg)):
                    run.events.append((t, 'STATE', f'{sm.group(1)} -> {sm.group(2)}'))
                elif (pm2 := START_POSE.search(msg)):
                    run.start = (to_float(pm2.group(2)), to_float(pm2.group(3)))
                    run.events.append((t, 'START', msg))
                elif msg.startswith(('Reporte', 'Vuelvo', 'Regreso', 'Vuelta terminada',
                                     'Se termina el tiempo')):
                    run.events.append((t, 'MISION', msg))
    return runs


def fmt_t(t):
    return '   -  ' if t is None else f'{t:6.1f}'


def still_periods(poses):
    """(t start, t end, x, y) where consecutive poses moved less than STILL_MOVE_M.

    AMCL publishes its pose only after the robot moves, so a long gap between pose lines is
    also a stop.
    """
    out = []
    anchor = None
    for t, x, y, _, _ in poses:
        if t is None or math.isnan(x):
            continue
        if anchor is None:
            anchor = (t, x, y, t)
            continue
        t0, x0, y0, _ = anchor
        if math.hypot(x - x0, y - y0) < STILL_MOVE_M:
            anchor = (t0, x0, y0, t)
        else:
            if t - t0 >= STILL_WARN_S:
                out.append((t0, t, x0, y0))
            anchor = (t, x, y, t)
    if anchor is not None and anchor[3] - anchor[0] >= STILL_WARN_S:   # still at the end
        out.append((anchor[0], anchor[3], anchor[1], anchor[2]))
    return out


def traveled(poses):
    d, last = 0.0, None
    for t, x, y, _, _ in poses:
        if math.isnan(x):
            continue
        if last is not None:
            d += math.hypot(x - last[0], y - last[1])
        last = (x, y)
    return d


def summary(run, out=sys.stdout):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f'Run: {run.header}')
    last = run.last_t
    p(f'Last line at t={fmt_t(last).strip()} s after /map'
      + ('' if run.ended else '  (no "run end" line: killed, crashed or still running)')
      + ('  LOG FULL' if run.full else ''))
    if run.map:
        w, h, res, ox, oy = run.map
        p(f'Map {w}x{h} cells, {res:.3f} m/cell = {w * res:.2f} x {h * res:.2f} m, '
          f'origin ({ox:.2f}, {oy:.2f})')
    else:
        p('No /map in this run')
    p(f'Reports: {len(run.reports)}   /done: {"yes at t=" + fmt_t(run.done[0]).strip() if run.done else "no"}'
      f'   path: {traveled(run.poses):.1f} m   pose lines: {len(run.poses)}')
    if run.reports:
        times = [t for t, *_ in run.reports if t is not None]
        if times:
            p(f'Last report at t={max(times):.1f} s (fast bonus: all N reports before 300 s)')
    if run.done and run.start and run.poses:
        t_done = run.done[0]
        before = [q for q in run.poses if q[0] is not None and t_done is not None and q[0] <= t_done + 1]
        if before:
            _, x, y, _, _ = before[-1]
            p(f'At /done the last AMCL pose is {math.hypot(x - run.start[0], y - run.start[1]):.3f} m '
              f'from mission\'s start pose (needs < 0.25 m, real error is larger)')

    p('\nTime line (t = s after /map)')
    for t, kind, text in sorted(run.events, key=lambda e: (e[0] is None, e[0] or 0)):
        p(f'  {fmt_t(t)}  {kind:<7} {text}')

    stops = still_periods(run.poses)
    p(f'\nStood still >= {STILL_WARN_S:.0f} s: {len(stops)}' + (' (judge ends the run at 60 s)' if stops else ''))
    for t0, t1, x, y in stops:
        flag = '  <-- 60 s' if t1 - t0 >= 60 else ''
        p(f'  t={t0:6.1f} .. {t1:6.1f} ({t1 - t0:5.1f} s) at ({x:.2f}, {y:.2f}){flag}')

    sds = [(q[4], q[0], q[1], q[2]) for q in run.poses if not math.isnan(q[4])]
    if sds:
        worst = sorted(sds, reverse=True)[:5]
        doubtful = sum(1 for s in sds if s[0] > DOUBT_SD_M)
        p(f'\nAMCL standard deviation: max {worst[0][0]:.3f} m, '
          f'{doubtful}/{len(sds)} pose lines above {DOUBT_SD_M} m')
        for sd, t, x, y in worst:
            p(f'  sd={sd:.3f} at t={fmt_t(t).strip()} ({x:.2f}, {y:.2f})')

    bad = [entry for entry in run.logs if entry[1] in ('WARN', 'ERROR', 'FATAL')]
    p(f'\nWARN/ERROR lines: {len(bad)}  (repeats within 5 s are folded by registro)')
    per_node = Counter((lvl, node) for _, lvl, node, _, _ in bad)
    for (lvl, node), n in per_node.most_common(12):
        p(f'  {n:5d}  {lvl:<5} {node}')
    common = Counter()
    first_t = {}
    for t, lvl, node, msg, rep in bad:
        key = (lvl, node, re.sub(r'-?\d+(\.\d+)?', '#', msg)[:120])
        common[key] += 1 + rep
        first_t.setdefault(key, t)
    if common:
        p('  most common (numbers as #), first time seen:')
        for (lvl, node, msg), n in common.most_common(10):
            p(f'  {n:5d}x  t={fmt_t(first_t[(lvl, node, msg)]).strip():>6}  [{node}] {msg}')


def diff_params(a, b, out=sys.stdout):
    p = lambda *x: print(*x, file=out)  # noqa: E731
    files = sorted(set(a.params) | set(b.params))
    changed = False
    for name in files:
        la, lb = a.params.get(name, []), b.params.get(name, [])
        strip = lambda ls: [l.split('#', 1)[0].rstrip() for l in ls if l.split('#', 1)[0].strip()]  # noqa: E731,E741
        sa, sb = strip(la), strip(lb)
        if sa == sb:
            continue
        changed = True
        import difflib
        p(f'--- {name}')
        for d in difflib.unified_diff(sa, sb, lineterm='', n=2):
            if d.startswith(('---', '+++')):
                continue
            p(d)
    if not changed:
        p('Same params in both runs')


def svg(run, path):
    """Path colored by time, reports (red), stops (blue), start (green), /done (black)."""
    pts = [(t, x, y) for t, x, y, _, _ in run.poses if not math.isnan(x)]
    xs = [x for _, x, _ in pts] + [r[1] for r in run.reports if not math.isnan(r[1])]
    ys = [y for _, _, y in pts] + [r[2] for r in run.reports if not math.isnan(r[2])]
    if run.map:
        w, h, res, ox, oy = run.map
        xs += [ox, ox + w * res]
        ys += [oy, oy + h * res]
    if run.start:
        xs.append(run.start[0])
        ys.append(run.start[1])
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x0, x1, y0, y1 = min(xs) - 0.3, max(xs) + 0.3, min(ys) - 0.3, max(ys) + 0.3
    scale = 700.0 / max(x1 - x0, y1 - y0)
    W, H = (x1 - x0) * scale, (y1 - y0) * scale
    X = lambda x: (x - x0) * scale  # noqa: E731
    Y = lambda y: (y1 - y) * scale  # noqa: E731  (y up, like the map)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H + 30:.0f}" '
         f'font-family="sans-serif" font-size="11"><rect width="100%" height="100%" fill="white"/>']
    for gx in range(math.ceil(x0), math.floor(x1) + 1):
        s.append(f'<line x1="{X(gx):.1f}" y1="0" x2="{X(gx):.1f}" y2="{H:.1f}" stroke="#eee"/>'
                 f'<text x="{X(gx) + 2:.1f}" y="{H - 2:.1f}" fill="#aaa">{gx}</text>')
    for gy in range(math.ceil(y0), math.floor(y1) + 1):
        s.append(f'<line x1="0" y1="{Y(gy):.1f}" x2="{W:.1f}" y2="{Y(gy):.1f}" stroke="#eee"/>'
                 f'<text x="2" y="{Y(gy) - 2:.1f}" fill="#aaa">{gy}</text>')
    if run.map:
        w, h, res, ox, oy = run.map
        s.append(f'<rect x="{X(ox):.1f}" y="{Y(oy + h * res):.1f}" width="{w * res * scale:.1f}" '
                 f'height="{h * res * scale:.1f}" fill="none" stroke="#999" stroke-dasharray="4 3"/>')
    t_max = max([t for t, _, _ in pts if t is not None] or [1.0]) or 1.0
    for (ta, xa, ya), (tb, xb, yb) in zip(pts, pts[1:]):
        f = min(max((tb or 0.0) / t_max, 0.0), 1.0)
        color = f'hsl({240 - 240 * f:.0f},80%,45%)'          # blue early -> red late
        s.append(f'<line x1="{X(xa):.1f}" y1="{Y(ya):.1f}" x2="{X(xb):.1f}" y2="{Y(yb):.1f}" '
                 f'stroke="{color}" stroke-width="2"/>')
    for t0, t1, x, y in still_periods(run.poses):
        r = 4 + (t1 - t0) / 5
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r:.1f}" fill="none" stroke="#06c"/>'
                 f'<text x="{X(x) + r + 2:.1f}" y="{Y(y):.1f}" fill="#06c">still {t1 - t0:.0f}s @{t0:.0f}</text>')
    if run.start:
        s.append(f'<circle cx="{X(run.start[0]):.1f}" cy="{Y(run.start[1]):.1f}" r="6" fill="#2a2"/>')
    for i, (t, x, y, _) in enumerate(run.reports, 1):
        if math.isnan(x):
            continue
        # 0.25 m and 0.5 m circles: the scoring distances around a report.
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{0.25 * scale:.1f}" fill="none" stroke="#d00" stroke-opacity="0.5"/>'
                 f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{0.5 * scale:.1f}" fill="none" stroke="#d00" stroke-opacity="0.2"/>'
                 f'<text x="{X(x) - 4:.1f}" y="{Y(y) + 4:.1f}" fill="#d00" font-weight="bold">X{i}</text>'
                 f'<text x="{X(x) + 8:.1f}" y="{Y(y) - 8:.1f}" fill="#d00">t={fmt_t(t).strip()}</text>')
    if run.done and pts:
        before = [q for q in pts if q[0] is not None and run.done[0] is not None and q[0] <= run.done[0] + 1]
        if before:
            _, x, y = before[-1]
            s.append(f'<rect x="{X(x) - 5:.1f}" y="{Y(y) - 5:.1f}" width="10" height="10" fill="black"/>')
    s.append(f'<text x="6" y="{H + 20:.0f}">{run.header} | path blue (early) to red (late) | '
             f'green start, X reports (0.25/0.5 m rings), black /done, blue still</text></svg>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--run', type=int, default=0, help='run number (1 = first, default: last)')
    ap.add_argument('--list', action='store_true', help='list the runs in the file')
    ap.add_argument('--svg', help='write a drawing of the run to this file')
    ap.add_argument('--diff', type=int, nargs=2, metavar=('A', 'B'), help='params changed from run A to B')
    ap.add_argument('--grep', help='print every line of the run that contains this text')
    ap.add_argument('--params', action='store_true', help='print the params the run used')
    args = ap.parse_args(argv)

    with open(args.log, encoding='utf-8', errors='replace') as f:
        runs = parse(f)
    if not runs:
        print('No run in this file (is it a registro.py log?)')
        return 1
    if args.list:
        for i, r in enumerate(runs, 1):
            print(f'{i:3d}  {r.header:<22} last t={fmt_t(r.last_t).strip():>6}  reports={len(r.reports)}'
                  f'  done={"yes" if r.done else "no"}  lines={len(r.lines)}')
        return 0
    if args.diff:
        a, b = (runs[i - 1] for i in args.diff)
        diff_params(a, b)
        return 0
    run = runs[args.run - 1] if args.run else runs[-1]
    if args.grep:
        for wall, t, text in run.lines:
            if args.grep in text:
                print(f'{wall} {fmt_t(t)} {text}')
        return 0
    if args.params:
        for name, lines in run.params.items():
            print(f'--- {name}')
            print('\n'.join(lines))
        return 0
    if len(runs) > 1:
        print(f'({len(runs)} runs in the file; showing {args.run or len(runs)}. --list shows them)\n')
    summary(run)
    if args.svg:
        svg(run, args.svg)
        print(f'\nDrawing: {args.svg} (open it in a browser)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
