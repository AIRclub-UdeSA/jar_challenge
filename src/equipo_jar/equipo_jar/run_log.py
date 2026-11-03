"""
The run's log file (docs/LOGGING.md): everything the team's processes print, in the single file
the organizers collect after each run. Runs inside the launch process; no node knows about it.

A failing log must never hurt the robot, and the rules don't let anyone restart the software.
So the event handlers only put a line on a bounded queue and never raise, and all the file work
is in a daemon thread: a full or stalled disk costs lines, not time. Launch reads the nodes'
pipes on its event loop, so waiting for the disk there would end up blocking the nodes.
"""
import atexit
import functools
import os
import queue
import sys
import threading
import time

FILE_NAME = 'log.txt'
DEFAULT_DIR = 'logs_jar'
ENV_DIR = 'JAR_LOG_DIR'
MAX_BYTES = 24_000_000      # the file may have 25 MB, whichever MB they mean
RESET_BYTES = 12_000_000    # a file over this at the start is emptied: room for this run
MAX_QUEUED = 20_000         # lines waiting for the disk; more are dropped
MAX_PARTIAL = 65_536        # bytes of a line without its end yet
RETRY_S = 5.0               # s between attempts to open the file after a failure
EXIT_WAIT_S = 1.0           # s for the writer to finish when launch exits
CUT_MARK = '=== log size limit reached, the rest of this run is not logged\n'


def log_dir(environ=None, prefix=None):
    """The folder of the log: $JAR_LOG_DIR, or logs_jar/ next to the workspace's install/.
    None if there is neither: nothing else on the disk may be written to."""
    environ = os.environ if environ is None else environ
    if environ.get(ENV_DIR):
        return environ[ENV_DIR]
    if prefix is None:
        from ament_index_python.packages import get_package_prefix
        prefix = get_package_prefix('equipo_jar')
    path = os.path.abspath(prefix)
    while os.path.basename(path) != 'install':
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    return os.path.join(os.path.dirname(path), DEFAULT_DIR)


def safe(method):
    """Never raise into launch: an exception in an event handler can end the launch, and every
    node with it."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            self.dropped += 1
    return wrapper


class RunLog:
    """Lines to the end of one file, from a thread of its own. Starts writing with the first
    line; without a folder (directory None) it takes lines and drops them."""

    def __init__(self, directory):
        self.directory = directory
        self.path = os.path.join(directory, FILE_NAME) if directory else None
        self.queue = queue.Queue(MAX_QUEUED)
        self.dropped = 0        # lines lost and not yet said so in the file
        self.partial = {}       # (process, fd) -> bytes of its unfinished line
        self.thread = None
        self.fd = None
        self.first_open = True
        self.next_open = 0.0
        self.cut = False        # hit MAX_BYTES: no more lines

    # On the launch process' event loop: never block, never raise.

    @safe
    def line(self, source, text):
        if self.path is None:
            return
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, daemon=True, name='run_log')
            self.thread.start()
            atexit.register(self.close)
        data = f'{time.time():.3f} [{source}] {text}\n'.encode('utf-8', 'replace')
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            self.dropped += 1

    @safe
    def header(self, arguments, params_path=None):
        """Starts a run in the file: when, the launch arguments, and the parameters it runs."""
        when = time.strftime('%Y-%m-%d %H:%M:%S %z')
        self.line('run_log', f'=== run start {when}, ' + ', '.join(
            f'{k}:={v}' for k, v in arguments.items()))
        if params_path:
            with open(params_path) as f:
                for text in f.read().splitlines():
                    self.line('params', text)

    @safe
    def on_output(self, event):
        """A chunk of a process' stdout or stderr: not whole lines, and may cut one in two."""
        key = (event.process_name, 1 if event.from_stdout else 2)
        data = self.partial.pop(key, b'') + event.text
        *lines, rest = data.split(b'\n')
        for text in lines:
            self.line(key[0], text.rstrip(b'\r').decode('utf-8', 'replace'))
        if len(rest) > MAX_PARTIAL:
            self.line(key[0], rest.decode('utf-8', 'replace'))
        elif rest:
            self.partial[key] = rest

    @safe
    def on_start(self, event, context=None):
        self.line(event.process_name, f'process started, pid {event.pid}')

    @safe
    def on_exit(self, event, context=None):
        for key in [k for k in self.partial if k[0] == event.process_name]:
            self.line(key[0], self.partial.pop(key).decode('utf-8', 'replace'))
        self.line(event.process_name,
                  f'process exited, pid {event.pid}, return code {event.returncode}')

    # The writer thread: the only code that touches the disk.

    def run(self):
        while True:
            batch = [self.queue.get()]
            try:
                while len(batch) < 1000:
                    batch.append(self.queue.get_nowait())
            except queue.Empty:
                pass
            closing = None in batch
            batch = [text for text in batch if text is not None]
            try:
                self.store(batch)
            except Exception:
                self.dropped += len(batch)
                self.close_file()
                self.next_open = time.monotonic() + RETRY_S
            if closing:
                self.close_file()
                return

    def store(self, batch):
        if not self.ready():
            self.dropped += len(batch)
            return
        if self.dropped and not self.cut:
            lost, self.dropped = self.dropped, 0
            batch.insert(0, f'{time.time():.3f} [run_log] === {lost} lines lost\n'.encode())
        size = os.fstat(self.fd).st_size
        room = MAX_BYTES - len(CUT_MARK) - size
        data = bytearray()
        for text in batch:
            if self.cut or len(text) > room:
                self.cut = True
                break
            data += text
            room -= len(text)
        if self.cut:
            data += CUT_MARK.encode()
        view = memoryview(data)
        while view:     # a write may take only part of it
            view = view[os.write(self.fd, view):]

    def ready(self):
        """Whether there is a file to write to, (re)opening it if it's not there anymore."""
        if self.cut:
            return False
        if self.fd is not None:
            try:
                if os.stat(self.path).st_ino == os.fstat(self.fd).st_ino:
                    return True
            except OSError:
                pass
            self.close_file()
        if time.monotonic() < self.next_open:
            return False
        try:
            os.makedirs(self.directory, exist_ok=True)
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            if self.first_open and os.fstat(self.fd).st_size > RESET_BYTES:
                os.ftruncate(self.fd, 0)
            self.first_open = False
        except OSError:
            self.close_file()
            self.next_open = time.monotonic() + RETRY_S
            return False
        return True

    def close_file(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def close(self):
        """At exit: what's queued gets written, if the disk answers in time."""
        for key in list(self.partial):
            self.line(key[0], self.partial.pop(key).decode('utf-8', 'replace'))
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            return
        self.thread.join(EXIT_WAIT_S)


def start():
    """The launch's RunLog. If there is no folder for it, the run goes on without a file."""
    directory = log_dir()
    if directory is None:
        print(f'[run_log] no {ENV_DIR} and no install/ in the package prefix: no log file',
              file=sys.stderr)
    return RunLog(directory)


def event_handlers(log):
    """The launch actions that send every process of the launch service to log."""
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnProcessExit, OnProcessIO, OnProcessStart
    return [
        RegisterEventHandler(OnProcessIO(on_stdout=log.on_output, on_stderr=log.on_output)),
        RegisterEventHandler(OnProcessStart(on_start=log.on_start)),
        RegisterEventHandler(OnProcessExit(on_exit=log.on_exit)),
    ]
