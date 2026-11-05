import threading

import rclpy

from equipo_jar.keep_alive import spin


def test_a_failing_callback_does_not_stop_the_node():
    rclpy.init()
    try:
        node = rclpy.create_node('keep_alive_test')
        calls = []

        def tick():
            calls.append(len(calls))
            if len(calls) == 1:
                raise RuntimeError('a bad message')
            if len(calls) == 3:
                rclpy.shutdown()

        node.create_timer(0.01, tick)
        t = threading.Thread(target=spin, args=(node,), daemon=True)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), 'spin should return on shutdown'
        assert len(calls) == 3, 'the timer should run again after it raised'
    finally:
        if rclpy.ok():
            rclpy.shutdown()
