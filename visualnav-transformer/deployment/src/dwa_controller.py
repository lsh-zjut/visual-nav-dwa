#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwa_controller.py

使用动态窗口法（Dynamic Window Approach，DWA）实现移动机器人在障碍环境中的局部路径规划。
本节点接收 navigate.py 发布的 waypoint 以及传感器信息，计算安全的线/角速度并发布到 /cmd_vel。
"""

import dataclasses
import math
import threading
from typing import Tuple, List

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool
import yaml

from ros_data import ROSData
from topic_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC
from utils import clip_angle


# ========================  配置与基本常量  ========================

CONFIG_PATH = "../config/robot.yaml"  # 机器人参数配置文件路径，与实际部署保持一致

with open(CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
    robot_cfg = yaml.safe_load(cfg_file)

# 速度与话题配置（若 robot.yaml 未提供则使用默认值）
MAX_V = float(robot_cfg.get("max_v", 0.25))                 # 最大线速度 (m/s)
MAX_W = float(robot_cfg.get("max_w", 0.30))                 # 最大角速度 (rad/s)
VEL_TOPIC = robot_cfg.get("vel_teleop_topic", "/cmd_vel")   # 速度指令发布话题
SCAN_TOPIC = robot_cfg.get("base_scan_topic", "/front/scan")  # 激光雷达话题
ODOM_TOPIC = robot_cfg.get("odom_topic", "/odometry/filtered")  # 里程计话题
ROBOT_RADIUS = float(robot_cfg.get("robot_radius", 0.25))   # 机器人等效半径 (m)，用于障碍膨胀

# 数据有效期，超时则视为无数据
WAYPOINT_TIMEOUT = 1.0  # waypoint 超时时间 (s)
LIDAR_TIMEOUT = 0.5     # 激光数据超时时间 (s)
ODOM_TIMEOUT = 0.5      # 里程计数据超时时间 (s)

# 动态窗口参数
DT = 0.1               # 控制周期，同时作为轨迹仿真步长 (s)
PRED_HORIZON = 10      # 轨迹预测步数，值越大考虑的未来越远
ACC_V = 0.4            # 线速度最大加速度 (m/s^2)
ACC_W = 2.5            # 角速度最大加速度 (rad/s^2)
RES_V = 0.02           # 线速度采样分辨率
RES_W = 0.02           # 角速度采样分辨率

# 评价函数权重
WEIGHT_HEADING = 3.2      # 朝向对齐权重
WEIGHT_VELOCITY = 0.15    # 保持速度权重
WEIGHT_CLEARANCE = 1.8    # 障碍间隙权重
WEIGHT_GOAL_DIST = 2.8    # waypoint 距离权重

CLEARANCE_HARD = 0.05     # 障碍硬约束，小于该距离直接淘汰 (m)
GOAL_DISTANCE_SCALE = 1.5 # waypoint 距离打分的指数衰减尺度 (m)
HEADING_LOCK_ANGLE = math.radians(25.0)  # 偏航超过该角度时降低线速度
HEADING_LOCK_DECAY = math.radians(60.0)  # 偏航误差的衰减区间
TURN_PENALTY_WEIGHT = 2.5  # 转弯惩罚权重，避免盲目高速转向
MIN_LINEAR_CMD = 0.15      # 最小前进速度指令 (m/s)，防止速度过小导致停滞


# ========================  状态结构体定义  ========================

@dataclasses.dataclass
class RobotState:
    """表示机器人在局部坐标系下的姿态和速度。"""
    x: float
    y: float
    yaw: float
    v: float
    w: float


# ========================  DWA 控制器实现  ========================

class DWAController:
    """基于 DWA 的局部规划控制节点。"""

    def __init__(self) -> None:
        rospy.init_node("DWA_CONTROLLER", anonymous=False)
        self._lock = threading.Lock()
        self._min_spin_speed = 0.05       # fallback 自转模式允许的最小角速度 (rad/s)
        self._preferred_spin_speed = 0.2  # fallback 自转模式期望角速度 (rad/s)

        # 使用 ROSData 封装订阅数据，自动处理缓存和超时
        self._waypoint = ROSData(timeout=WAYPOINT_TIMEOUT, name="waypoint")
        self._scan = ROSData(timeout=LIDAR_TIMEOUT, name="scan")
        self._odom = ROSData(timeout=ODOM_TIMEOUT, name="odom")
        self._goal_reached = False

        # 订阅 waypoint / 终点标志 / 激光雷达 / 里程计
        rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self._goal_cb, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=1)

        # 速度命令发布者
        self._cmd_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
        self._rate = rospy.Rate(int(1.0 / DT))

    # --------------------  回调函数  --------------------

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        """接收局部 waypoint（相对机器人坐标系），写入缓存。"""
        self._waypoint.set(msg.data)

    def _goal_cb(self, msg: Bool) -> None:
        """接收上游导航是否到达终点的标志。"""
        self._goal_reached = msg.data

    def _scan_cb(self, msg: LaserScan) -> None:
        """接收激光雷达扫描数据。"""
        self._scan.set(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """接收里程计估计的机器人位姿与速度。"""
        self._odom.set(msg)

    # --------------------  主循环  --------------------

    def run(self) -> None:
        """循环读取传感器数据，计算最优速度，并发布到 /cmd_vel。"""
        rospy.loginfo("DWA controller is ready.")
        while not rospy.is_shutdown():
            twist = Twist()

            # 已到达目标则保持原地
            if self._goal_reached:
                self._cmd_pub.publish(twist)
                rospy.loginfo_throttle(1.0, "Goal reached flag set. Holding still.")
                self._rate.sleep()
                continue

            # 数据不齐全则等待
            if not self._data_ready():
                self._cmd_pub.publish(twist)
                self._rate.sleep()
                continue

            # 读取当前状态并整理 waypoint（仅取平面 x, y）
            state = self._get_state()
            waypoint = np.array(self._waypoint.get(), dtype=np.float32)
            waypoint = waypoint[:2] if waypoint.shape[0] >= 2 else waypoint

            # 调用 DWA 寻优，得到线速度与角速度
            best_v, best_w = self._plan(state, waypoint, self._scan.get())
            if best_v > 0.0 and best_v < MIN_LINEAR_CMD:
                best_v = MIN_LINEAR_CMD

            twist.linear.x = best_v
            twist.angular.z = best_w

            rospy.loginfo("输出速度 -> v: %.3f m/s, w: %.3f rad/s", best_v, best_w)
            self._cmd_pub.publish(twist)
            self._rate.sleep()

    # --------------------  数据判断与状态转换  --------------------

    def _data_ready(self) -> bool:
        """确认 waypoint / scan / odom 均在有效期内。"""
        waypoint_ok = self._waypoint.is_valid(verbose=True)
        scan_ok = self._scan.is_valid(verbose=True)
        odom_ok = self._odom.is_valid(verbose=True)

        if not waypoint_ok:
            rospy.logwarn_throttle(1.0, "等待 waypoint 数据...")
        if not scan_ok:
            rospy.logwarn_throttle(1.0, "等待激光雷达 /front/scan 数据...")
        if not odom_ok:
            rospy.logwarn_throttle(1.0, "等待里程计数据...")

        return waypoint_ok and scan_ok and odom_ok

    def _get_state(self) -> RobotState:
        """从里程计消息提取机器人在世界系下的位姿和当前速度。"""
        odom: Odometry = self._odom.get()
        pose = odom.pose.pose
        twist = odom.twist.twist

        yaw = self._quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        return RobotState(
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw,
            v=twist.linear.x,
            w=twist.angular.z,
        )

    # --------------------  DWA 采样与评分  --------------------

    def _plan(self, state: RobotState, waypoint: np.ndarray, scan: LaserScan) -> Tuple[float, float]:
        """根据当前状态和传感器数据，遍历动态窗口内的速度候选并选择最优解。"""
        dynamic_window = self._compute_dynamic_window(state)
        rospy.logdebug(
            "动态窗口: v=[%.3f, %.3f], w=[%.3f, %.3f]",
            dynamic_window[0],
            dynamic_window[1],
            dynamic_window[2],
            dynamic_window[3],
        )

        best_score = -float("inf")
        best_cmd = (0.0, 0.0)

        ranges = np.array(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            rospy.logwarn_throttle(1.0, "激光数据为空，尝试原地旋转重新搜索路径。")
            return self._fallback_spin(dynamic_window, ranges, scan.range_min, scan.range_max)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))

        for v in np.arange(dynamic_window[0], dynamic_window[1] + 1e-5, RES_V):
            for w in np.arange(dynamic_window[2], dynamic_window[3] + 1e-5, RES_W):
                traj = self._simulate(state, v, w)
                terminal_state = traj[-1]

                heading_error = abs(self._heading_error(terminal_state, waypoint))
                heading_cost = self._heading_cost(terminal_state, waypoint)
                goal_tracking_cost = self._goal_distance_cost(terminal_state, waypoint)
                velocity_cost = max(v, 0.0) / max(MAX_V, 1e-3)
                clearance_cost = self._clearance_cost(
                    traj, ranges, angles, scan.range_min, scan.range_max
                )

                # 不满足安全距离则直接淘汰
                if clearance_cost < CLEARANCE_HARD:
                    continue

                if heading_error > HEADING_LOCK_ANGLE:
                    decay = max(
                        0.0,
                        1.0 - (heading_error - HEADING_LOCK_ANGLE) / max(HEADING_LOCK_DECAY, 1e-6),
                    )
                    velocity_cost *= decay

                alignment_penalty = (heading_error / math.pi) * (v / max(MAX_V, 1e-3))

                score = (
                    WEIGHT_HEADING * heading_cost
                    + WEIGHT_VELOCITY * velocity_cost
                    + WEIGHT_CLEARANCE * clearance_cost
                    + WEIGHT_GOAL_DIST * goal_tracking_cost
                    - TURN_PENALTY_WEIGHT * alignment_penalty
                )

                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        if best_score == -float("inf"):
            rospy.logwarn_throttle(1.0, "未找到可行速度候选，尝试原地旋转。")
            spin_cmd = self._fallback_spin(dynamic_window, ranges, scan.range_min, scan.range_max)
            if abs(spin_cmd[1]) > 1e-3:
                rospy.loginfo("进入自转模式搜索路径，角速度 w=%.3f", spin_cmd[1])
            else:
                rospy.logwarn_throttle(1.0, "自转也受限，继续保持原地等待。")
            return spin_cmd

        rospy.logdebug("最优评分 %.3f (v=%.3f, w=%.3f)", best_score, best_cmd[0], best_cmd[1])

        return best_cmd

    # --------------------  辅助函数  --------------------

    def _compute_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """结合当前速度与加速度约束，计算下一周期可达的速度范围。"""
        v_min = max(0.0, state.v - ACC_V * DT)
        v_max = min(MAX_V, state.v + ACC_V * DT)
        w_min = max(-MAX_W, state.w - ACC_W * DT)
        w_max = min(MAX_W, state.w + ACC_W * DT)
        return v_min, v_max, w_min, w_max

    def _simulate(self, state: RobotState, v: float, w: float) -> List[RobotState]:
        """在机器人局部坐标系中仿真未来轨迹，返回每一步的状态。"""
        traj: List[RobotState] = []
        x, y, yaw = 0.0, 0.0, 0.0  # 以机器人当前位置作为原点

        for _ in range(PRED_HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            traj.append(RobotState(x, y, yaw, v, w))

        return traj

    def _heading_error(self, state: RobotState, waypoint: np.ndarray) -> float:
        """计算终点朝向与 waypoint 方位的角度差（弧度）。"""
        if waypoint.shape[0] < 2:
            return 0.0
        goal_heading = math.atan2(waypoint[1], waypoint[0])
        return clip_angle(goal_heading - state.yaw)

    def _heading_cost(self, state: RobotState, waypoint: np.ndarray) -> float:
        """朝向越接近 waypoint 得分越高，范围 [0, 1]。"""
        angle_diff = abs(self._heading_error(state, waypoint))
        return 1.0 - angle_diff / math.pi

    def _goal_distance_cost(self, state: RobotState, waypoint: np.ndarray) -> float:
        """终点越接近 waypoint 得分越高，范围 [0, 1]。"""
        if waypoint.shape[0] < 2:
            return 0.0
        dx = waypoint[0] - state.x
        dy = waypoint[1] - state.y
        distance = math.hypot(dx, dy)
        scale = max(GOAL_DISTANCE_SCALE, 1e-3)
        return math.exp(-distance / scale)

    def _clearance_cost(
        self,
        traj: List[RobotState],
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> float:
        """
        遍历轨迹上的点，统计最小障碍间隙，减去机器人半径后归一化到 [0, 1]。
        1 表示安全裕量大，0 表示碰撞或穿模。
        """
        min_clearance = range_max
        effective_max = max(range_max - ROBOT_RADIUS, range_min)

        for point in traj:
            px, py = point.x, point.y
            dist = math.hypot(px, py)
            heading = math.atan2(py, px)
            idx = int((heading - angles[0]) / (angles[-1] - angles[0]) * (len(angles) - 1))

            if 0 <= idx < len(ranges):
                obstacle_dist = ranges[idx]
                if not np.isfinite(obstacle_dist):
                    obstacle_dist = range_max

                # 膨胀障碍物以考虑机器人半径
                obstacle_dist = max(obstacle_dist - ROBOT_RADIUS, range_min)

                if obstacle_dist > range_min:
                    margin = max(obstacle_dist - dist, 0.0)
                    min_clearance = min(min_clearance, margin)

        min_clearance = max(min_clearance, 0.0)
        denominator = max(effective_max, 1e-3)
        return min(min_clearance / denominator, 1.0)

    def _fallback_spin(
        self,
        dynamic_window: Tuple[float, float, float, float],
        ranges: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> Tuple[float, float]:
        """无可行速度时保持原地，根据障碍分布选择安全方向自转。"""
        v_spin = 0.0
        max_left = max(dynamic_window[3], 0.0)
        max_right = max(-dynamic_window[2], 0.0)
        spin_limit = max(max_left, max_right)
        if spin_limit < 1e-3:
            spin_limit = MAX_W

        spin_mag = min(self._preferred_spin_speed, spin_limit)
        if spin_mag < self._min_spin_speed:
            spin_mag = spin_limit
        if spin_mag < self._min_spin_speed:
            return v_spin, 0.0

        direction = 1.0
        if ranges.size > 0:
            mid = ranges.size // 2
            left_score = self._direction_clearance(ranges[mid:], range_min, range_max)
            right_score = self._direction_clearance(ranges[:mid], range_min, range_max)
            if right_score > left_score:
                direction = -1.0

        spin_cmd = direction * spin_mag
        spin_cmd = max(min(spin_cmd, dynamic_window[3]), dynamic_window[2])

        if abs(spin_cmd) < self._min_spin_speed:
            spin_cmd = self._min_spin_speed * (1.0 if spin_cmd >= 0.0 else -1.0)
            spin_cmd = max(min(spin_cmd, dynamic_window[3]), dynamic_window[2])

        if abs(spin_cmd) < 1e-3:
            return v_spin, 0.0

        return v_spin, spin_cmd

    @staticmethod
    def _direction_clearance(
        samples: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> float:
        """统计一组激光距离的平均可行间隙，用于比较左右方向的通行性。"""
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
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """四元数转换为 yaw（围绕 Z 轴的偏航角）。"""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


# ========================  节点入口  ========================

def main() -> None:
    controller = DWAController()
    controller.run()


if __name__ == "__main__":
    main()
