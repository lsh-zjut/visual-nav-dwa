#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwa_relative_controller.py

仅依赖 waypoint (Float32MultiArray) 的机器人坐标系 Δx、Δy 指令和激光雷达，
使用动态窗口法（DWA）生成安全的线速度/角速度，并发布到 /cmd_vel。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool


# ========================  参数配置  ========================

WAYPOINT_TOPIC = "/waypoint"
SCAN_TOPIC = "/front/scan"
CMD_TOPIC = "/cmd_vel"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

WAYPOINT_TIMEOUT = 2   # waypoint 超时时间 (s)
LIDAR_TIMEOUT = 0.5      # 激光雷达超时时间 (s)
CMD_TIMEOUT = 1.0        # 若长时间收不到 waypoint，则停下 (s)

DT = 0.1                 # 控制周期 (s)
PRED_HORIZON = 10        # 轨迹预测步数
ACC_V = 0.4              # 线速度最大加速度 (m/s^2)
ACC_W = 5.0              # 角速度最大加速度 (rad/s^2)
MAX_V = 0.3              # 线速度上限 (m/s)
MAX_W = 0.5              # 角速度上限 (rad/s)
RES_V = 0.02             # 线速度采样间隔
RES_W = 0.02             # 角速度采样间隔

WEIGHT_HEADING = 4.0
WEIGHT_VELOCITY = 0.2
WEIGHT_CLEARANCE = 1.8
WEIGHT_GOAL_DIST = 2.6
TURN_PENALTY_WEIGHT = 1

CLEARANCE_HARD = 0.15    # 最低安全裕度 (m)
GOAL_DISTANCE_SCALE = 1.2
MIN_LINEAR_CMD = 0.05    # 防止速度过小导致卡滞

REACHED_GOAL_RADIUS = 0.05  # waypoint 距离阈值 (m)
ROBOT_RADIUS = 0.25         # 用于 clearance 估计 (m)


# ========================  数据结构  ========================

@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    w: float


# ========================  主体实现  ========================

class DWAController:
    def __init__(self) -> None:
        rospy.init_node("DWA_RELATIVE_CONTROLLER", anonymous=False)

        self._last_waypoint = None
        self._last_waypoint_time = None
        self._last_scan = None
        self._last_scan_time = None

        self._goal_reached = False

        self._current_v = 0.0
        self._current_w = 0.0

        rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self._goal_cb, queue_size=1)

        self._cmd_pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        self._rate = rospy.Rate(int(1.0 / DT))

        rospy.loginfo("DWA (relative) controller ready.")

    # -------------------- 回调 --------------------

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            rospy.logwarn_throttle(2.0, "收到的 waypoint 少于 2 个元素，忽略。")
            return
        self._last_waypoint = np.array(msg.data[:2], dtype=np.float32)
        self._last_waypoint_time = rospy.Time.now()

    def _scan_cb(self, msg: LaserScan) -> None:
        self._last_scan = msg
        self._last_scan_time = rospy.Time.now()

    def _goal_cb(self, msg: Bool) -> None:
        self._goal_reached = msg.data

    # -------------------- 主循环 --------------------

    def run(self) -> None:
        while not rospy.is_shutdown():
            twist = Twist()

            if self._goal_reached:
                self._current_v = 0.0
                self._current_w = 0.0
                self._cmd_pub.publish(twist)
                rospy.loginfo_throttle(1.0, "外部标志：已到达目标，保持静止。")
                self._rate.sleep()
                continue

            if not self._data_ready():
                self._current_v = 0.0
                self._current_w = 0.0
                self._cmd_pub.publish(twist)
                self._rate.sleep()
                continue

            waypoint_rel = self._last_waypoint.copy()
            if np.linalg.norm(waypoint_rel) < REACHED_GOAL_RADIUS:
                self._current_v = 0.0
                self._current_w = 0.0
                self._cmd_pub.publish(twist)
                rospy.loginfo_throttle(1.0, "Wayoint 距离过近，保持原地等待。")
                self._rate.sleep()
                continue

            state = RobotState(
                x=0.0,
                y=0.0,
                yaw=0.0,
                v=self._current_v,
                w=self._current_w,
            )

            best_v, best_w = self._plan(state, waypoint_rel, self._last_scan)
            if 0.0 < best_v < MIN_LINEAR_CMD:
                best_v = MIN_LINEAR_CMD

            twist.linear.x = best_v
            twist.angular.z = best_w

            self._current_v = best_v
            self._current_w = best_w

            rospy.loginfo(
                "发送速度: v = %.3f m/s, w = %.3f rad/s (目标 Δx=%.2f, Δy=%.2f)",
                best_v,
                best_w,
                waypoint_rel[0],
                waypoint_rel[1],
            )

            self._cmd_pub.publish(twist)
            self._rate.sleep()

    # -------------------- 数据校验 --------------------

    def _data_ready(self) -> bool:
        now = rospy.Time.now()

        if (
            self._last_waypoint is None
            or self._last_waypoint_time is None
            or (now - self._last_waypoint_time).to_sec() > WAYPOINT_TIMEOUT
        ):
            rospy.logwarn_throttle(1.0, "等待新 waypoint...")
            return False

        if (
            self._last_scan is None
            or self._last_scan_time is None
            or (now - self._last_scan_time).to_sec() > LIDAR_TIMEOUT
        ):
            rospy.logwarn_throttle(1.0, f"等待激光雷达 {SCAN_TOPIC} 数据...")
            return False

        return True

    # -------------------- DWA 规划 --------------------

    def _plan(self, state: RobotState, waypoint_rel: np.ndarray, scan: LaserScan) -> Tuple[float, float]:
        dynamic_window = self._compute_dynamic_window(state)

        best_score = -float("inf")
        best_cmd = (0.0, 0.0)

        ranges = np.array(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            rospy.logwarn_throttle(1.0, "激光数据为空，采用原地旋转。")
            return self._fallback_spin(dynamic_window, ranges, scan.range_min, scan.range_max)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        for v in np.arange(dynamic_window[0], dynamic_window[1] + 1e-5, RES_V):
            for w in np.arange(dynamic_window[2], dynamic_window[3] + 1e-5, RES_W):
                traj = self._simulate(state, v, w)
                terminal_state = traj[-1]

                heading_error = abs(self._heading_error(terminal_state, waypoint_rel))
                heading_cost = self._heading_cost(terminal_state, waypoint_rel)
                goal_cost = self._goal_distance_cost(terminal_state, waypoint_rel)
                velocity_cost = max(v, 0.0) / max(MAX_V, 1e-3)
                clearance_cost = self._clearance_cost(
                    traj, ranges, angles, scan.range_min, scan.range_max
                )

                if clearance_cost < CLEARANCE_HARD:
                    continue

                if heading_error > math.radians(25.0):
                    decay = max(
                        0.0,
                        1.0 - (heading_error - math.radians(25.0)) / math.radians(60.0),
                    )
                    velocity_cost *= decay

                alignment_penalty = (heading_error / math.pi) * (v / max(MAX_V, 1e-3))

                score = (
                    WEIGHT_HEADING * heading_cost
                    + WEIGHT_VELOCITY * velocity_cost
                    + WEIGHT_CLEARANCE * clearance_cost
                    + WEIGHT_GOAL_DIST * goal_cost
                    - TURN_PENALTY_WEIGHT * alignment_penalty
                )

                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        if best_score == -float("inf"):
            rospy.logwarn_throttle(1.0, "未找到可行速度，尝试原地旋转。")
            return self._fallback_spin(dynamic_window, ranges, scan.range_min, scan.range_max)

        return best_cmd

    # -------------------- 动态窗口、轨迹与代价 --------------------

    def _compute_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        v_min = max(0.0, state.v - ACC_V * DT)
        v_max = min(MAX_V, state.v + ACC_V * DT)
        w_min = max(-MAX_W, state.w - ACC_W * DT)
        w_max = min(MAX_W, state.w + ACC_W * DT)
        return v_min, v_max, w_min, w_max

    def _simulate(self, state: RobotState, v: float, w: float) -> List[RobotState]:
        traj: List[RobotState] = []
        x, y, yaw = 0.0, 0.0, 0.0

        for _ in range(PRED_HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            traj.append(RobotState(x, y, yaw, v, w))

        return traj

    def _heading_error(self, state: RobotState, waypoint_rel: np.ndarray) -> float:
        goal_heading = math.atan2(waypoint_rel[1], waypoint_rel[0])
        return self._wrap_angle(goal_heading - state.yaw)

    def _heading_cost(self, state: RobotState, waypoint_rel: np.ndarray) -> float:
        angle_diff = abs(self._heading_error(state, waypoint_rel))
        return 1.0 - angle_diff / math.pi

    def _goal_distance_cost(self, state: RobotState, waypoint_rel: np.ndarray) -> float:
        dx = waypoint_rel[0] - state.x
        dy = waypoint_rel[1] - state.y
        distance = math.hypot(dx, dy)
        return math.exp(-distance / max(GOAL_DISTANCE_SCALE, 1e-3))

    def _clearance_cost(
        self,
        traj: List[RobotState],
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> float:
        min_clearance = range_max
        effective_max = max(range_max - ROBOT_RADIUS, range_min)

        for point in traj:
            dist = math.hypot(point.x, point.y)
            heading = math.atan2(point.y, point.x)
            idx = int((heading - angles[0]) / (angles[-1] - angles[0]) * (len(angles) - 1))

            if 0 <= idx < len(ranges):
                obstacle_dist = ranges[idx]
                if not np.isfinite(obstacle_dist):
                    obstacle_dist = range_max

                obstacle_dist = max(obstacle_dist - ROBOT_RADIUS, range_min)
                if obstacle_dist > range_min:
                    margin = max(obstacle_dist - dist, 0.0)
                    min_clearance = min(min_clearance, margin)

        min_clearance = max(min_clearance, 0.0)
        return min(min_clearance / max(effective_max, 1e-3), 1.0)

    def _fallback_spin(
        self,
        dynamic_window: Tuple[float, float, float, float],
        ranges: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> Tuple[float, float]:
        v_spin = 0.0
        max_left = max(dynamic_window[3], 0.0)
        max_right = max(-dynamic_window[2], 0.0)
        spin_limit = max(max_left, max_right, 1e-3)

        spin_speed = min(0.25, spin_limit)
        if spin_speed < 0.08:
            spin_speed = spin_limit

        direction = 1.0
        if ranges.size > 0:
            mid = ranges.size // 2
            left_score = self._direction_clearance(ranges[mid:], range_min, range_max)
            right_score = self._direction_clearance(ranges[:mid], range_min, range_max)
            if right_score > left_score:
                direction = -1.0

        spin_cmd = direction * spin_speed
        spin_cmd = max(min(spin_cmd, dynamic_window[3]), dynamic_window[2])

        if abs(spin_cmd) < 0.05:
            spin_cmd = 0.05 * (1 if spin_cmd >= 0 else -1)

        return v_spin, spin_cmd

    @staticmethod
    def _direction_clearance(samples: np.ndarray, range_min: float, range_max: float) -> float:
        if samples.size == 0:
            return range_min
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return range_max
        adjusted = np.clip(finite - ROBOT_RADIUS, 0.0, range_max)
        if adjusted.size == 0:
            return range_min
        return float(np.mean(adjusted))

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2 * math.pi) - math.pi


# ========================  入口函数  ========================

def main() -> None:
    controller = DWAController()
    controller.run()


if __name__ == "__main__":
    main()
