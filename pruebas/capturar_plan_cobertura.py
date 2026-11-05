"""Capture a plan from a running test launch and draw its observation goals.

Only subscribes; it never publishes maps, TF, goals, velocities or reports.
Images and runtime coordinates go in ignored build output, never in source.
"""
import argparse
import json
import math
from pathlib import Path as FilePath
import time

from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.node import Node

from equipo_jar.comun import QOS_ESTADO, QOS_MAPA


def save_plot(grid, plan, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    import numpy as np

    origin = grid.info.origin
    q = origin.orientation
    yaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
    fig, ax = plt.subplots(figsize=(9, 8))
    transform = Affine2D().rotate(yaw).translate(origin.position.x, origin.position.y)
    data = np.array(grid.data).reshape(grid.info.height, grid.info.width)
    width, height = grid.info.width * grid.info.resolution, grid.info.height * grid.info.resolution
    ax.imshow(data, origin='lower', cmap='gray_r', vmin=0, vmax=100,
              extent=(0, width, 0, height), transform=transform + ax.transData)
    corners = transform.transform([(0, 0), (width, 0), (0, height), (width, height)])
    ax.set_xlim(min(corners[:, 0]), max(corners[:, 0]))
    ax.set_ylim(min(corners[:, 1]), max(corners[:, 1]))
    for index, stamped in enumerate(plan.poses):
        pose, q = stamped.pose, stamped.pose.orientation
        heading = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
        ax.arrow(pose.position.x, pose.position.y, 0.18 * math.cos(heading),
                 0.18 * math.sin(heading), width=0.012, head_width=0.07,
                 color='#087f8c', length_includes_head=True)
        ax.text(pose.position.x + 0.025, pose.position.y + 0.035, str(index + 1),
                fontsize=7, color='#9c2b10')
    ax.set(title=f'Initial coverage plan: {len(plan.poses)} observation goals',
           xlabel='map x [m]', ylabel='map y [m]', aspect='equal')
    fig.text(0.5, 0.02, 'Arrows show viewing direction. Navigation paths and detections are not shown.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='build/verificacion/cobertura/plan.png')
    parser.add_argument('--timeout', type=float, default=90.0)
    args = parser.parse_args()
    rclpy.init()
    node = Node('coverage_capture')
    received = {}
    node.create_subscription(OccupancyGrid, '/map', lambda msg: received.update(map=msg), QOS_MAPA)
    node.create_subscription(Path, '/jar/plan_cobertura',
                             lambda msg: received.update(plan=msg), QOS_ESTADO)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and not {'map', 'plan'} <= received.keys():
            rclpy.spin_once(node, timeout_sec=0.1)
        if not {'map', 'plan'} <= received.keys():
            raise RuntimeError('Timed out waiting for both /map and /jar/plan_cobertura')
        if not received['plan'].poses:
            raise RuntimeError('Coverage published an empty plan')
        destination = FilePath(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_plot(received['map'], received['plan'], destination)
        destination.with_suffix('.json').write_text(json.dumps({
            'frame': received['plan'].header.frame_id,
            'poses': [{'x': msg.pose.position.x, 'y': msg.pose.position.y,
                       'yaw': math.atan2(2 * msg.pose.orientation.w * msg.pose.orientation.z,
                                         1 - 2 * msg.pose.orientation.z ** 2)}
                      for msg in received['plan'].poses],
        }, indent=2) + '\n')
        print(f'Captured {len(received["plan"].poses)} viewpoints: {destination}', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
