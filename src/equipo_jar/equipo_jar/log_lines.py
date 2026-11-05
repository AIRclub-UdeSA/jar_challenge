"""
What the nodes log for the run's log (docs/LOGGING.md) that no decision of theirs prints: state
to reconstruct a run with afterwards. Only formatting, so it can be tested without ROS.
"""
import base64
import functools
import math
import zlib

import numpy as np

P_CHANGE = 0.1      # how much an object's P must move to get its own line
MAP_CHUNK = 1000    # characters of the encoded map per line


def guarded(method):
    """A node's logging step that can't hurt the node: if it fails, say so once and go on."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            try:
                if method.__name__ not in self.__dict__.setdefault('_log_failed', set()):
                    self._log_failed.add(method.__name__)
                    self.get_logger().error(f'{method.__name__} failed: {e!r}')
            except Exception:
                pass    # nor can saying so
    return wrapper


def sigma(obj):
    """m, the larger standard deviation of an object's center."""
    return math.sqrt(max(obj['cov'][0], obj['cov'][2], 0.0))


def object_fields(obj):
    x, y = obj['center']
    return (f'{obj["id"]} {x:.2f} {y:.2f} {obj["p"]:.3f} {obj["p_rojo"]:.3f} '
            f'{sigma(obj):.3f}')


def objects_line(objects):
    """All the objects: id x y P P(red) sigma for each."""
    return 'objects: ' + ('; '.join(object_fields(o) for o in objects) if objects else 'none')


def object_changes(objects, last):
    """Lines for the objects that appeared, are gone, or whose P moved P_CHANGE or more since
    their last line. last (id -> that P) is updated."""
    lines = []
    ids = set()
    for o in objects:
        ids.add(o['id'])
        before = last.get(o['id'])
        if before is None:
            lines.append(f'object new: {object_fields(o)}')
            last[o['id']] = o['p']
        elif abs(o['p'] - before) >= P_CHANGE:
            lines.append(f'object p: {before:.3f} -> {object_fields(o)}')
            last[o['id']] = o['p']
    for oid in sorted(set(last) - ids):
        lines.append(f'object gone: {oid}, last P {last.pop(oid):.3f}')
    return lines


def tour_line(name, tour):
    """A tour, in order: kind, object id (-1: none), x, y, value, estimated s, score per stop."""
    return f'{name}: ' + '; '.join(
        f'{t.kind} {t.object_id} {t.xy[0]:.2f} {t.xy[1]:.2f} {t.value:.2f} {t.time:.0f} '
        f'{t.score:.3f}' for t in tour)


def path_line(poses, t0):
    """The poses (stamp s, x, y, yaw) since the last line, in one: t since t0, x, y, yaw each."""
    return 'path: ' + '; '.join(
        f'{t - t0:.2f} {x:.3f} {y:.3f} {yaw:.3f}' for t, x, y, yaw in poses)


def map_lines(msg):
    """The map as log lines, to draw a run on: base64 of the zlib of its int8 cells, row 0 first."""
    i = msg.info
    data = np.asarray(msg.data, np.int8).tobytes()
    text = base64.b64encode(zlib.compress(data, 6)).decode()
    chunks = [text[k:k + MAP_CHUNK] for k in range(0, len(text), MAP_CHUNK)]
    return [f'map: {i.width}x{i.height} @ {i.resolution} m, origin ({i.origin.position.x}, '
            f'{i.origin.position.y}), {len(chunks)} parts'] + [
        f'map[{k + 1}/{len(chunks)}]: {c}' for k, c in enumerate(chunks)]


def decode_map(lines):
    """The cells (int8 array, height x width) back from the lines of map_lines, for tools."""
    size = next(s for s in lines if s.startswith('map: ')).split()[1]
    width, height = (int(v) for v in size.split('x'))
    text = ''.join(s.split(': ', 1)[1] for s in lines if s.startswith('map['))
    cells = np.frombuffer(zlib.decompress(base64.b64decode(text)), np.int8)
    return cells.reshape(height, width)
