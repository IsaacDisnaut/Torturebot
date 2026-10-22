#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class DriveDistance(Node):

    def __init__(self):
        super().__init__('drive_distance')

        # Parameters
        self.declare_parameter('target_distance', 2.0)
        self.declare_parameter('speed', 0.15)
        self.declare_parameter('odom_timeout', 1.0)

        self.target_distance = float(
            self.get_parameter('target_distance').value
        )

        self.speed = float(
            self.get_parameter('speed').value
        )

        self.odom_timeout = float(
            self.get_parameter('odom_timeout').value
        )

        # Validate parameters
        if self.target_distance <= 0.0:
            raise ValueError(
                'target_distance must be greater than 0'
            )

        if self.speed <= 0.0:
            raise ValueError(
                'speed must be greater than 0'
            )

        if self.odom_timeout <= 0.0:
            raise ValueError(
                'odom_timeout must be greater than 0'
            )

        # Odometry variables
        self.previous_x = None
        self.previous_y = None

        self.start_x = None
        self.start_y = None

        self.distance_traveled = 0.0
        self.last_odom_time = None

        self.started = False
        self.finished = False

        # Publisher: /cmd_vel
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # Subscriber: /odom
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Control loop: 20 Hz
        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.get_logger().info(
            'Drive Distance Controller started'
        )

        self.get_logger().info(
            f'Target distance: {self.target_distance:.3f} m'
        )

        self.get_logger().info(
            f'Speed: {self.speed:.3f} m/s'
        )

        self.get_logger().info(
            'Waiting for Odometry...'
        )

    def odom_callback(self, msg):

        # Current position from Odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.last_odom_time = time.monotonic()

        # Initialize starting position
        if self.previous_x is None:

            self.previous_x = x
            self.previous_y = y

            self.start_x = x
            self.start_y = y

            self.get_logger().info(
                f'Start position: x={x:.3f}, y={y:.3f}'
            )

            return

        # Accumulate distance between consecutive odom samples
        dx = x - self.previous_x
        dy = y - self.previous_y

        delta_distance = math.hypot(dx, dy)

        self.distance_traveled += delta_distance

        # Update previous position
        self.previous_x = x
        self.previous_y = y

    def publish_velocity(self, velocity):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.twist.linear.x = float(velocity)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.cmd_pub.publish(msg)

    def control_loop(self):

        # Do nothing until the first odometry message arrives
        if self.last_odom_time is None:
            return

        # Safety: stop if odometry becomes stale
        odom_age = time.monotonic() - self.last_odom_time

        if odom_age > self.odom_timeout:

            self.publish_velocity(0.0)

            if not self.finished:
                self.get_logger().error(
                    'Odometry timeout! Stopping robot.'
                )

                self.finished = True

            return

        # Start motion once the initial position is known
        if not self.started:

            self.started = True

            self.get_logger().info(
                'Starting robot movement...'
            )

        # Stop if target distance is reached
        if self.distance_traveled >= self.target_distance:

            self.publish_velocity(0.0)

            if not self.finished:

                self.finished = True

                self.get_logger().info(
                    'Target distance reached!'
                )

                self.get_logger().info(
                    f'Target: {self.target_distance:.3f} m'
                )

                self.get_logger().info(
                    f'Actual odom distance: '
                    f'{self.distance_traveled:.3f} m'
                )

                self.get_logger().info(
                    f'Final position: '
                    f'x={self.previous_x:.3f}, '
                    f'y={self.previous_y:.3f}'
                )

            return

        # Command forward velocity
        if not self.finished:

            self.publish_velocity(self.speed)

            remaining = (
                self.target_distance - self.distance_traveled
            )

            self.get_logger().info(
                f'Distance: {self.distance_traveled:.3f} m | '
                f'Remaining: {remaining:.3f} m',
                throttle_duration_sec=0.5
            )

    def stop_robot(self):

        # Send multiple zero-velocity commands before exiting
        for _ in range(5):
            self.publish_velocity(0.0)

            # Allow ROS to transmit the message
            time.sleep(0.05)

        self.get_logger().info(
            'Robot stop command sent.'
        )


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = DriveDistance()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except (ValueError, RuntimeError) as exc:
        print(f'Error: {exc}')

    finally:
        if node is not None:
            node.stop_robot()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()