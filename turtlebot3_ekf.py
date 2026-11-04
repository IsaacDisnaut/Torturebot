#!/usr/bin/env python3
"""
TurtleBot3 wheel-speed PID + EKF sensor fusion (encoder + IMU).

Pipeline
--------
  cmd_vel_ref (setpoint, TwistStamped)
        |
        v  inverse kinematics -> wheel setpoints [wL_ref, wR_ref] (rad/s)
  PID per wheel (feedback = wheel speed from /joint_states)
        |
        v  forward kinematics -> corrected Twist
  /cmd_vel  ------------------------------->  robot / Gazebo
        |
        +--> EKF predict (control input u = [v_cmd, w_cmd])
                 update #1: encoder  z = [v_enc, w_enc]
                 update #2: IMU gyro z = [w_imu]
        |
        v
  /odom_ekf (nav_msgs/Odometry)  (+ optional TF odom -> base_footprint)

EKF state: X = [x, y, theta, v, w]
Motion model (first-order actuator lag tau toward the commanded velocity):
  x'     = x + v*cos(theta)*dt
  y'     = y + v*sin(theta)*dt
  theta' = theta + w*dt
  v'     = v + a*(v_cmd - v),   a = dt/tau
  w'     = w + a*(w_cmd - w)

Usage
-----
  ros2 run turtlebot3_example turtlebot3_ekf
  # send setpoints to /cmd_vel_ref instead of /cmd_vel, e.g.
  ros2 run turtlebot3_teleop teleop_keyboard --ros-args -r cmd_vel:=cmd_vel_ref
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_to_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class PID:
    """Discrete PID with derivative-on-measurement and anti-windup clamp."""

    def __init__(self, kp, ki, kd, i_limit, out_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement

        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))

        derivative = 0.0
        if self.prev_meas is not None and dt > 0.0:
            derivative = -(measurement - self.prev_meas) / dt
        self.prev_meas = measurement

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.out_limit, min(self.out_limit, out))


class EKF:
    """EKF for a differential-drive robot. State: [x, y, theta, v, w]."""

    def __init__(self, q_diag, tau):
        self.x = np.zeros(5)
        self.P = np.eye(5) * 1e-3
        self.Q = np.diag(q_diag)
        self.tau = tau

    def predict(self, u, dt):
        v_cmd, w_cmd = u
        px, py, th, v, w = self.x
        a = min(1.0, dt / self.tau) if self.tau > 0.0 else 1.0

        c, s = math.cos(th), math.sin(th)
        self.x = np.array([
            px + v * c * dt,
            py + v * s * dt,
            wrap_angle(th + w * dt),
            v + a * (v_cmd - v),
            w + a * (w_cmd - w),
        ])

        F = np.eye(5)
        F[0, 2] = -v * s * dt
        F[0, 3] = c * dt
        F[1, 2] = v * c * dt
        F[1, 3] = s * dt
        F[2, 4] = dt
        F[3, 3] = 1.0 - a
        F[4, 4] = 1.0 - a

        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        # Joseph form for numerical stability
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


class Turtlebot3EKF(Node):

    def __init__(self):
        super().__init__('turtlebot3_ekf')

        # ---------------- Parameters ----------------
        p = self.declare_parameter
        p('wheel_radius', 0.033)
        p('wheel_separation', 0.160)          # burger 0.160, waffle 0.287
        p('left_joint', 'wheel_left_joint')
        p('right_joint', 'wheel_right_joint')
        p('rate', 50.0)                       # control/EKF rate (Hz)
        p('setpoint_topic', 'cmd_vel_ref')
        p('cmd_vel_topic', 'cmd_vel')
        p('odom_topic', 'odom_ekf')
        p('odom_frame', 'odom')
        p('base_frame', 'base_footprint')
        p('publish_tf', False)                # Gazebo already publishes odom TF
        p('setpoint_timeout', 0.5)            # s, stop if no setpoint
        p('enable_pid', True)
        # PID gains (output = wheel-speed correction, rad/s)
        p('kp', 0.8)
        p('ki', 2.0)
        p('kd', 0.0)
        p('i_limit', 2.0)
        p('max_wheel_speed', 6.67)            # rad/s (~0.22 m/s burger)
        # EKF tuning
        p('actuator_tau', 0.1)                # s
        p('q_diag', [1e-4, 1e-4, 1e-4, 0.5, 1.0])
        p('r_enc_v', 1e-4)
        p('r_enc_w', 4e-3)
        p('r_imu_w', 1e-4)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.r = float(g('wheel_radius'))
        self.L = float(g('wheel_separation'))
        self.left_joint = g('left_joint')
        self.right_joint = g('right_joint')
        self.dt_nominal = 1.0 / float(g('rate'))
        self.odom_frame = g('odom_frame')
        self.base_frame = g('base_frame')
        self.publish_tf = bool(g('publish_tf'))
        self.setpoint_timeout = float(g('setpoint_timeout'))
        self.enable_pid = bool(g('enable_pid'))
        self.max_wheel = float(g('max_wheel_speed'))

        kp, ki, kd = float(g('kp')), float(g('ki')), float(g('kd'))
        i_lim = float(g('i_limit'))
        self.pid_l = PID(kp, ki, kd, i_lim, self.max_wheel)
        self.pid_r = PID(kp, ki, kd, i_lim, self.max_wheel)

        self.ekf = EKF([float(q) for q in g('q_diag')], float(g('actuator_tau')))
        self.R_enc = np.diag([float(g('r_enc_v')), float(g('r_enc_w'))])
        self.R_imu = np.array([[float(g('r_imu_w'))]])
        self.H_enc = np.array([[0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]], dtype=float)
        self.H_imu = np.array([[0, 0, 0, 0, 1]], dtype=float)

        # ---------------- State ----------------
        self.v_ref = 0.0
        self.w_ref = 0.0
        self.last_setpoint_time = None

        self.wl_meas = 0.0
        self.wr_meas = 0.0
        self.enc_new = False
        self.prev_joint_pos = None
        self.prev_joint_time = None

        self.imu_wz = 0.0
        self.imu_new = False

        self.last_loop_time = None

        # ---------------- ROS I/O ----------------
        self.cmd_pub = self.create_publisher(TwistStamped, g('cmd_vel_topic'), 10)
        self.odom_pub = self.create_publisher(Odometry, g('odom_topic'), 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.create_subscription(
            TwistStamped, g('setpoint_topic'), self.setpoint_callback, 10)
        self.create_subscription(
            JointState, 'joint_states', self.joint_callback, qos_profile_sensor_data)
        self.create_subscription(
            Imu, 'imu', self.imu_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(self.dt_nominal, self.loop)

        self.get_logger().info(
            f"turtlebot3_ekf started: setpoint='{g('setpoint_topic')}' -> "
            f"PID -> '{g('cmd_vel_topic')}', EKF -> '{g('odom_topic')}'")

    # ---------------- Kinematics ----------------
    def twist_to_wheels(self, v, w):
        wl = (v - w * self.L / 2.0) / self.r
        wr = (v + w * self.L / 2.0) / self.r
        return wl, wr

    def wheels_to_twist(self, wl, wr):
        v = self.r * (wr + wl) / 2.0
        w = self.r * (wr - wl) / self.L
        return v, w

    # ---------------- Callbacks ----------------
    def setpoint_callback(self, msg):
        self.v_ref = msg.twist.linear.x
        self.w_ref = msg.twist.angular.z
        self.last_setpoint_time = self.get_clock().now()

    def joint_callback(self, msg):
        try:
            il = msg.name.index(self.left_joint)
            ir = msg.name.index(self.right_joint)
        except ValueError:
            return

        if len(msg.velocity) > max(il, ir):
            self.wl_meas = msg.velocity[il]
            self.wr_meas = msg.velocity[ir]
            self.enc_new = True
            return

        # Fallback: differentiate encoder positions
        if len(msg.position) <= max(il, ir):
            return
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pos = (msg.position[il], msg.position[ir])
        if self.prev_joint_pos is not None:
            dt = t - self.prev_joint_time
            if dt > 1e-4:
                self.wl_meas = (pos[0] - self.prev_joint_pos[0]) / dt
                self.wr_meas = (pos[1] - self.prev_joint_pos[1]) / dt
                self.enc_new = True
        self.prev_joint_pos = pos
        self.prev_joint_time = t

    def imu_callback(self, msg):
        self.imu_wz = msg.angular_velocity.z
        self.imu_new = True

    # ---------------- Main loop ----------------
    def loop(self):
        now = self.get_clock().now()
        if self.last_loop_time is None:
            self.last_loop_time = now
            return
        dt = (now - self.last_loop_time).nanoseconds * 1e-9
        self.last_loop_time = now
        if dt <= 0.0:
            return
        dt = min(dt, 5.0 * self.dt_nominal)

        # Setpoint watchdog
        if (self.last_setpoint_time is None or
                (now - self.last_setpoint_time).nanoseconds * 1e-9 > self.setpoint_timeout):
            self.v_ref, self.w_ref = 0.0, 0.0

        # ---- 1) PID on wheel speeds ----
        wl_ref, wr_ref = self.twist_to_wheels(self.v_ref, self.w_ref)
        idle = abs(wl_ref) < 1e-6 and abs(wr_ref) < 1e-6

        if idle:
            self.pid_l.reset()
            self.pid_r.reset()
            wl_cmd, wr_cmd = 0.0, 0.0
        elif self.enable_pid:
            # feed-forward + PID correction
            wl_cmd = wl_ref + self.pid_l.update(wl_ref, self.wl_meas, dt)
            wr_cmd = wr_ref + self.pid_r.update(wr_ref, self.wr_meas, dt)
        else:
            wl_cmd, wr_cmd = wl_ref, wr_ref

        wl_cmd = max(-self.max_wheel, min(self.max_wheel, wl_cmd))
        wr_cmd = max(-self.max_wheel, min(self.max_wheel, wr_cmd))
        v_cmd, w_cmd = self.wheels_to_twist(wl_cmd, wr_cmd)
        self.publish_cmd(v_cmd, w_cmd, now)

        # ---- 2) EKF predict with PID output as control input ----
        self.ekf.predict((v_cmd, w_cmd), dt)

        # ---- 3) EKF update: encoder ----
        if self.enc_new:
            v_enc, w_enc = self.wheels_to_twist(self.wl_meas, self.wr_meas)
            self.ekf.update(np.array([v_enc, w_enc]), self.H_enc, self.R_enc)
            self.enc_new = False

        # ---- 4) EKF update: IMU gyro ----
        if self.imu_new:
            self.ekf.update(np.array([self.imu_wz]), self.H_imu, self.R_imu)
            self.imu_new = False

        # ---- 5) Publish odometry ----
        self.publish_odom(now)

    # ---------------- Publishers ----------------
    def publish_cmd(self, v, w, stamp):
        msg = TwistStamped()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.base_frame
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def publish_odom(self, stamp):
        x, y, th, v, w = self.ekf.x
        P = self.ekf.P
        qx, qy, qz, qw = yaw_to_quaternion(th)

        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(v)
        odom.twist.twist.angular.z = float(w)

        # 6x6 covariance [x y z roll pitch yaw]; map EKF indices (x,y,theta)
        pose_cov = np.zeros((6, 6))
        idx = [0, 1, 5]
        for i, si in enumerate([0, 1, 2]):
            for j, sj in enumerate([0, 1, 2]):
                pose_cov[idx[i], idx[j]] = P[si, sj]
        pose_cov[2, 2] = pose_cov[3, 3] = pose_cov[4, 4] = 1e6  # unused DOF
        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = P[3, 3]
        twist_cov[0, 5] = twist_cov[5, 0] = P[3, 4]
        twist_cov[5, 5] = P[4, 4]
        twist_cov[1, 1] = twist_cov[2, 2] = 1e6
        twist_cov[3, 3] = twist_cov[4, 4] = 1e6
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header = odom.header
            t.child_frame_id = self.base_frame
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

    def stop(self):
        self.publish_cmd(0.0, 0.0, self.get_clock().now())


def main(args=None):
    rclpy.init(args=args)
    node = Turtlebot3EKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
