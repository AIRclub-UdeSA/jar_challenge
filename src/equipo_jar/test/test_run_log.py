import errno
import os
import stat
import time
from types import SimpleNamespace

import numpy as np

from equipo_jar import log_lines, run_log
from equipo_jar.run_log import RunLog, log_dir


def output(text, process='mission-4', fd=2):
    return SimpleNamespace(process_name=process, from_stdout=fd == 1, text=text)


def finish(log):
    log.close()
    assert not log.thread.is_alive()


def lines_of(directory):
    with open(os.path.join(directory, 'log.txt')) as f:
        return f.read().splitlines()


def wait_until(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


# The folder.

def test_the_variable_wins_when_it_has_a_value():
    prefix = '/ws/install/equipo_jar'
    assert log_dir({'JAR_LOG_DIR': '/x/y'}, prefix) == '/x/y'
    assert log_dir({'JAR_LOG_DIR': ''}, prefix) == '/ws/logs_jar'
    assert log_dir({}, prefix) == '/ws/logs_jar'


def test_merged_install_and_no_install():
    assert log_dir({}, '/ws/install') == '/ws/logs_jar'
    assert log_dir({}, '/opt/somewhere/equipo_jar') is None


def test_the_symlink_install_counts_where_it_is_not_where_it_points(tmp_path):
    real = tmp_path / 'build' / 'equipo_jar'
    real.mkdir(parents=True)
    (tmp_path / 'install').mkdir()
    link = tmp_path / 'install' / 'equipo_jar'
    link.symlink_to(real)
    assert log_dir({}, str(link)) == str(tmp_path / 'logs_jar')


# Writing.

def test_creates_the_folder_and_writes_lines_with_their_process(tmp_path):
    d = tmp_path / 'a' / 'b'
    log = RunLog(str(d))
    log.on_output(output(b'[INFO] one\n[INFO] two\n'))
    finish(log)
    lines = lines_of(d)
    assert [s.split(' ', 1)[1] for s in lines] == ['[mission-4] [INFO] one', '[mission-4] [INFO] two']


def test_a_line_cut_in_two_chunks_is_one_line_per_process(tmp_path):
    log = RunLog(str(tmp_path))
    log.on_output(output(b'half of '))
    log.on_output(output(b'other\n', process='motion-3'))
    log.on_output(output(b'a line\r\nrest without end'))
    finish(log)
    assert [s.split(' ', 1)[1] for s in lines_of(tmp_path)] == [
        '[motion-3] other', '[mission-4] half of a line', '[mission-4] rest without end']


def test_bad_bytes_and_bad_events_do_not_raise(tmp_path):
    log = RunLog(str(tmp_path))
    log.on_output(output(b'\xff\xfe bad\n'))
    log.on_output(object())
    log.on_start(object(), None)
    log.on_exit(object(), None)
    log.header({'victimas': 3}, '/nonexistent/params.yaml')
    finish(log)
    assert any('bad' in s for s in lines_of(tmp_path))


def test_process_start_and_exit_with_return_code(tmp_path):
    log = RunLog(str(tmp_path))
    log.on_start(SimpleNamespace(process_name='motion-3', pid=7), None)
    log.on_output(output(b'no end', process='motion-3'))
    log.on_exit(SimpleNamespace(process_name='motion-3', pid=7, returncode=-11), None)
    finish(log)
    text = '\n'.join(lines_of(tmp_path))
    assert 'process started, pid 7' in text
    assert text.index('no end') < text.index('return code -11')


def test_a_new_run_is_added_to_the_file(tmp_path):
    (tmp_path / 'log.txt').write_text('the run before\n')
    log = RunLog(str(tmp_path))
    log.header({'victimas': 3})
    finish(log)
    lines = lines_of(tmp_path)
    assert lines[0] == 'the run before'
    assert '=== run start' in lines[1] and 'victimas:=3' in lines[1]


def test_a_big_file_is_emptied_at_the_start_only(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, 'RESET_BYTES', 100)
    (tmp_path / 'log.txt').write_text('x' * 101 + '\n')
    log = RunLog(str(tmp_path))
    for k in range(50):
        log.line('t', f'line {k} ' + 'y' * 10)
    finish(log)
    lines = lines_of(tmp_path)
    assert len(lines) == 50 and 'xxx' not in lines[0]


def test_a_file_at_the_limit_stays_under_it_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, 'MAX_BYTES', 2000)
    log = RunLog(str(tmp_path))
    for k in range(500):
        log.line('t', f'line {k}')
    finish(log)
    assert os.path.getsize(tmp_path / 'log.txt') <= 2000
    assert lines_of(tmp_path)[-1] == run_log.CUT_MARK.strip()


# Failing.

def test_no_folder_takes_lines_and_writes_nothing():
    log = RunLog(None)
    log.line('t', 'x')
    log.on_output(output(b'y\n'))
    assert log.thread is None


def test_a_folder_that_cannot_be_created_only_costs_lines(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('')
    log = RunLog(str(blocked / 'sub'))
    log.line('t', 'x')
    finish(log)
    assert log.dropped == 1


def test_a_read_only_folder_only_costs_lines(tmp_path):
    d = tmp_path / 'ro'
    d.mkdir()
    d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    log = RunLog(str(d))
    log.line('t', 'x')
    finish(log)
    d.chmod(stat.S_IRWXU)
    assert not (d / 'log.txt').exists() or os.geteuid() == 0


def test_log_txt_being_a_folder_only_costs_lines(tmp_path):
    (tmp_path / 'log.txt').mkdir()
    log = RunLog(str(tmp_path))
    log.line('t', 'x')
    finish(log)
    assert log.dropped == 1


def test_a_full_disk_costs_lines_and_the_file_says_how_many(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, 'RETRY_S', 0.0)
    real = os.write
    full = [True]

    def write(fd, data):
        if full[0]:
            raise OSError(errno.ENOSPC, 'no space')
        return real(fd, data)

    monkeypatch.setattr(run_log.os, 'write', write)
    log = RunLog(str(tmp_path))
    log.line('t', 'lost')
    assert wait_until(lambda: log.dropped >= 1)
    full[0] = False
    log.line('t', 'kept')
    finish(log)
    text = '\n'.join(lines_of(tmp_path))
    assert 'lines lost' in text and 'kept' in text and '[t] lost' not in text


def test_a_stalled_disk_never_makes_the_handlers_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, 'MAX_QUEUED', 100)
    monkeypatch.setattr(run_log, 'EXIT_WAIT_S', 0.05)
    monkeypatch.setattr(run_log.os, 'write', lambda fd, data: time.sleep(3600))
    log = RunLog(str(tmp_path))
    t0 = time.monotonic()
    for _ in range(5000):
        log.on_output(output(b'a line\n'))
    assert time.monotonic() - t0 < 1.0
    assert log.dropped > 0
    log.close()     # returns, with the writer still stuck
    assert log.thread.is_alive() and log.thread.daemon


def test_the_folder_deleted_during_the_run_comes_back(tmp_path):
    d = tmp_path / 'logs'
    log = RunLog(str(d))
    log.line('t', 'before')
    assert wait_until((d / 'log.txt').exists)
    assert wait_until(lambda: (d / 'log.txt').stat().st_size > 0)
    (d / 'log.txt').unlink()
    d.rmdir()
    log.line('t', 'after')
    finish(log)
    assert any('after' in s for s in lines_of(d))


# The launch actions.

def test_event_handlers_are_launch_actions():
    from launch.actions import RegisterEventHandler
    actions = run_log.event_handlers(RunLog(None))
    assert len(actions) == 3 and all(isinstance(a, RegisterEventHandler) for a in actions)


# What the nodes log.

def obj(oid, p, x=1.0, y=2.0):
    return {'id': oid, 'center': (x, y), 'p': p, 'p_rojo': 0.5, 'cov': (0.04, 0.0, 0.01)}


def test_objects_line():
    assert log_lines.objects_line([]) == 'objects: none'
    assert log_lines.objects_line([obj(3, 0.25)]) == 'objects: 3 1.00 2.00 0.250 0.500 0.200'


def test_object_changes():
    last = {}
    assert log_lines.object_changes([obj(1, 0.2)], last) == [
        'object new: 1 1.00 2.00 0.200 0.500 0.200']
    assert log_lines.object_changes([obj(1, 0.29)], last) == []
    # Small steps add up: measured from the last line, not the last snapshot.
    assert log_lines.object_changes([obj(1, 0.31)], last) == [
        'object p: 0.200 -> 1 1.00 2.00 0.310 0.500 0.200']
    assert log_lines.object_changes([], last) == ['object gone: 1, last P 0.310']
    assert last == {}


def test_tour_line():
    stops = [SimpleNamespace(kind='look', object_id=20, xy=(2.561, 4.78), value=0.32, time=8.4,
                             score=0.0391),
             SimpleNamespace(kind='explore', object_id=-1, xy=(1.0, -2.0), value=0.1, time=20.0,
                             score=0.005)]
    assert log_lines.tour_line('tour', stops) == (
        'tour: look 20 2.56 4.78 0.32 8 0.039; explore -1 1.00 -2.00 0.10 20 0.005')


def test_path_line():
    poses = [(100.5, 1.0, 2.0, 0.5), (100.6, 1.0126, -2.0, -3.14159)]
    assert log_lines.path_line(poses, 100.0) == (
        'path: 0.50 1.000 2.000 0.500; 0.60 1.013 -2.000 -3.142')


def test_the_map_comes_back_from_its_lines():
    cells = np.random.default_rng(0).choice([-1, 0, 100], 240 * 200).astype(np.int8)
    msg = SimpleNamespace(data=cells.tolist(), info=SimpleNamespace(
        width=240, height=200, resolution=0.03,
        origin=SimpleNamespace(position=SimpleNamespace(x=-1.5, y=-2.0))))
    lines = log_lines.map_lines(msg)
    assert lines[0].startswith('map: 240x200 @ 0.03 m, origin (-1.5, -2.0), ')
    assert all(len(s) < 1100 for s in lines)
    assert (log_lines.decode_map(lines) == cells.reshape(200, 240)).all()


def test_a_failing_log_step_says_so_once_and_does_not_raise():
    said = []

    class Node:
        def get_logger(self):
            return SimpleNamespace(error=said.append)

        @log_lines.guarded
        def step(self):
            raise RuntimeError('bug')

    node = Node()
    node.step()
    node.step()
    assert len(said) == 1


def test_a_log_step_without_a_logger_does_not_raise():
    class Node:
        def get_logger(self):
            raise AttributeError('no logger')

        @log_lines.guarded
        def step(self):
            raise RuntimeError('bug')

    Node().step()
