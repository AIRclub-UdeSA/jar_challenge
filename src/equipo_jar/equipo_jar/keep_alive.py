"""
Spinning that survives a failing callback.

rclpy.spin re-raises a callback's exception and the node's process ends: one bad message
would stop the robot for the rest of the run, and the rules don't let anyone restart the
software (§11.4). spin() logs the exception and keeps spinning, so the next message gets its
chance. Only a shutdown (Ctrl-C, the launch ending) ends it.
"""
import traceback

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor


def spin(node, executor=None):
    """Spin node (on executor, or a new single-threaded one) until shutdown, logging the
    callbacks that raise: the first few, then one in 100."""
    if executor is None:
        executor = SingleThreadedExecutor()
        executor.add_node(node)
    errors = 0
    while rclpy.ok():
        try:
            executor.spin()
        except (KeyboardInterrupt, ExternalShutdownException):
            return
        except Exception as e:  # never die mid-run: log and keep spinning
            errors += 1
            if errors <= 5 or errors % 100 == 0:
                node.get_logger().error(f'callback failed ({errors}): {e!r}\n'
                                        + traceback.format_exc())
