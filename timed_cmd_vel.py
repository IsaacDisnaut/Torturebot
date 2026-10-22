#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class TimedCmdVel(Node):

    def __init__(self):
        super().__init__('timed_cmd_vel')

        # Parameters
        self.declare_parameter('linear_speed', 0.2)
        self.declare_parameter('angular_speed', 0.0)
        self.declare_parameter('duration', 10.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('frame_id', 'base_link')

        self.linear_speed = float(
            self.get_parameter('linear_speed').value
        )
        self.angular_speed = float(
            self.get_parameter('angular_speed').value
        )
        self.duration = float(
            self.get_parameter('duration').value
        )
        self.publish_rate = float(
            self.get_parameter('publish_rate').value
        )
        self.frame_id = str(
            self.get_parameter('frame_id').value
        )

        # State
        self.start_time = None
        self.finished = False

        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Timer
        timer_period = 1.0 / max(self.publish_rate, 0.1)

        self.timer = self.create_timer(
            timer_period,
            self.control_loop
        )

        self.get_logger().info('Timed Cmd Vel started')

    def publish_velocity(self, linear_x, angular_z):
        """Publish TwistStamped velocity command."""

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(angular_z)

        self.cmd_vel_pub.publish(msg)

    def control_loop(self):
        """Publish velocity and count down immediately."""

        if self.finished:
            return

        # Start the timer immediately on the first control cycle.
        if self.start_time is None:
            self.start_time = time.monotonic()

            self.get_logger().info(
                f'Command started: linear={self.linear_speed} m/s, '
                f'angular={self.angular_speed} rad/s'
            )
            self.get_logger().info(
                f'Counting down {self.duration:.2f} seconds'
            )

        elapsed = time.monotonic() - self.start_time

        if elapsed < self.duration:
            # Keep publishing the target velocity.
            self.publish_velocity(
                self.linear_speed,
                self.angular_speed
            )

        else:
            # Stop after the duration has elapsed.
            self.publish_velocity(0.0, 0.0)

            self.finished = True

            self.get_logger().info(
                f'Duration completed: {self.duration:.2f} seconds'
            )
            self.get_logger().info(
                'Zero velocity published. Robot stopped.'
            )

    def destroy_node(self):
        """Publish zero velocity when shutting down."""

        for _ in range(5):
            self.publish_velocity(0.0, 0.0)
            time.sleep(0.05)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TimedCmdVel()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()