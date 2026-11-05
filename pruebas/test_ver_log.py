"""ver_log.py reads registro.py's log format: runs, time line, stops, params diff, drawing."""
import importlib.util
import io
from pathlib import Path

spec = importlib.util.spec_from_file_location('ver_log', Path(__file__).with_name('ver_log.py'))
ver_log = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ver_log)


def line(t, text, wall='12:00:00'):
    stamp = '   -   ' if t is None else f'{t:7.1f}'
    return f'{wall} {stamp} {text}\n'


def run_lines(speed_param):
    out = [line(None, '===== run start 2026-11-04 10:00:00 ====='),
           line(None, '----- params/navegacion.yaml -----'),
           line(None, 'amcl:'),
           line(None, '  ros__parameters:'),
           line(None, f'    max_speed: {speed_param}   # comment'),
           line(None, '----- end of params -----'),
           line(None, 'INFO  [amcl] startup line'),
           line(0.0, 'MAP 120x100 res 0.050 origin (-0.50, -0.60)'),
           line(0.5, 'INFO  [mision] Pose de inicio fija: x=0.00 y=0.00 yaw=0.00'),
           line(0.6, 'INFO  [mision] Estado: <ESPERA> -> <EXPLORACION>')]
    # moves to (1, 0), stands still 25 s, moves to (2, 0), stands still 65 s
    t = 1.0
    for i in range(11):
        out.append(line(t, f'POSE {0.1 * i:.3f} 0.000 0.000 sd_xy 0.020 0.030'))
        t += 1.0
    for _ in range(25):
        out.append(line(t, 'POSE 1.005 0.001 0.000 sd_xy 0.200 0.030'))
        t += 1.0
    out.append(line(t, 'POSE 2.000 0.000 0.000 sd_xy 0.020 0.030'))
    out.append(line(t, 'REPORT 2.100 0.500 frame map'))
    out.append(line(t, 'WARN  [controller_server] Control loop missed its desired rate of 20.0000Hz [+12 same]'))
    out.append(line(t + 65.0, 'POSE 2.001 0.000 0.000 sd_xy 0.020 0.030'))
    out.append(line(t + 66.0, 'DONE'))
    out.append(line(t + 67.0, 'ERROR [planner_server] Failed to create plan with tolerance of: 0.500000'))
    out.append(line(t + 68.0, '===== run end ====='))
    return out


def parsed():
    return ver_log.parse(run_lines(0.3) + run_lines(0.4))


def test_splits_runs_and_reads_events():
    runs = parsed()
    assert len(runs) == 2
    run = runs[0]
    assert run.map == (120, 100, 0.05, -0.5, -0.6)
    assert run.start == (0.0, 0.0)
    assert run.params['navegacion.yaml'][-1].strip().startswith('max_speed: 0.3')
    assert [r[1:3] for r in run.reports] == [(2.1, 0.5)]
    assert run.done and run.ended and not run.full
    kinds = [k for _, k, _ in run.events]
    assert kinds[:3] == ['MAP', 'START', 'STATE'] and 'DONE' in kinds
    assert len(run.logs) == 5 and run.logs[3][4] == 12      # folded repeats are read


def test_finds_long_stops():
    stops = ver_log.still_periods(parsed()[0].poses)
    lengths = [round(t1 - t0) for t0, t1, _, _ in stops]
    assert lengths == [26, 65]      # until the next pose that moved
    assert abs(ver_log.traveled(parsed()[0].poses) - 2.0) < 0.02


def test_summary_mentions_what_matters():
    out = io.StringIO()
    ver_log.summary(parsed()[1], out)
    text = out.getvalue()
    for piece in ('Reports: 1', '/done: yes', '<-- 60 s', 'sd=0.200', '[planner_server]',
                  'Control loop missed its desired rate of #Hz', 'STATE   ESPERA -> EXPLORACION'):
        assert piece in text, piece


def test_params_diff_ignores_comments():
    runs = parsed()
    out = io.StringIO()
    ver_log.diff_params(runs[0], runs[1], out)
    assert '-    max_speed: 0.3' in out.getvalue() and '+    max_speed: 0.4' in out.getvalue()
    out = io.StringIO()
    ver_log.diff_params(runs[0], runs[0], out)
    assert 'Same params' in out.getvalue()


def test_cli_and_drawing(tmp_path, capsys):
    log = tmp_path / 'log.txt'
    log.write_text(''.join(run_lines(0.3)))
    drawing = tmp_path / 'run.svg'
    assert ver_log.main([str(log), '--svg', str(drawing)]) == 0
    assert drawing.read_text().startswith('<svg') and 'X1' in drawing.read_text()
    assert ver_log.main([str(log), '--list']) == 0
    assert ver_log.main([str(log), '--grep', 'Failed']) == 0
    assert 'Failed to create plan' in capsys.readouterr().out


def test_survives_garbage_and_cut_files(tmp_path):
    lines = run_lines(0.3)
    cut = lines[:20] + ['garbage line\n', '12:00:00    1.0 POSE nan nan nan sd_xy nan nan\n',
                        lines[20][:15]]
    runs = ver_log.parse(cut)
    assert len(runs) == 1 and not runs[0].ended
    ver_log.summary(runs[0], io.StringIO())
    ver_log.svg(runs[0], tmp_path / 'cut.svg')
    assert ver_log.parse(['no header line\n']) == []
