#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dwa_controller.py

使用动态窗口法（Dynamic Window Approach, DWA）实现基于激光雷达的局部避障控制器。
节点订阅模型产生的局部 waypoint、里程计和激光雷达数据，在 ROS 中发布 /cmd_vel 控制小车运动。
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


# ========================  配置读取与常量定义  ========================

CONFIG_PATH = "../config/robot.yaml"  # 机器人配置文件路径（运行时相对本脚本）

with open(CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
    robot_cfg = yaml.safe_load(cfg_file)

# 速度与话题配置（可在 robot.yaml 中覆盖）
MAX_V = float(robot_cfg.get("max_v", 0.25))                          # 最大线速度 (m/s)
MAX_W = float(robot_cfg.get("max_w", 0.5))                          # 最大角速度 (rad/s)
VEL_TOPIC = robot_cfg.get("vel_teleop_topic", "/cmd_vel")            # 速度发布话题
SCAN_TOPIC = robot_cfg.get("base_scan_topic", "/front/scan")         # 激光雷达话题
ODOM_TOPIC = robot_cfg.get("odom_topic", "/odometry/filtered")       # 里程计话题
ROBOT_RADIUS = float(robot_cfg.get("robot_radius", 0.25))            # 机器人半径 (m)，用于避障膨胀

# 传感器数据有效期，超过该时间未更新则视为无效
WAYPOINT_TIMEOUT = 1.0  # waypoint 超时时间 (s)
LIDAR_TIMEOUT = 0.5     # 激光雷达超时时间 (s)
ODOM_TIMEOUT = 0.5      # 里程计超时时间 (s)

# 动态窗口参数
DT = 0.1               # 控制周期 (s)，同时是轨迹预测的时间分辨率
PRED_HORIZON = 10      # 轨迹预测步数，越大越远，代价是计算量增加
ACC_V = 0.4            # 线速度最大加速度 (m/s^2)
ACC_W = 2.5            # 角速度最大加速度 (rad/s^2)
RES_V = 0.02           # 线速度采样分辨率
RES_W = 0.02           # 角速度采样分辨率

# 评分函数权重（根据场景调节贴轨/避障/速度的倾向）
WEIGHT_HEADING = 3.2      # 朝向评分权重
WEIGHT_VELOCITY = 0.15     # 速度评分权重
WEIGHT_CLEARANCE = 1.8     # 障碍余量评分权重
WEIGHT_GOAL_DIST = 2.8     # waypoint 跟踪评分权重

CLEARANCE_HARD = 0.05      # 障碍物硬约束: 小于该余量直接丢弃候选 (m)
GOAL_DISTANCE_SCALE = 1.5  # waypoint 距离得分的指数衰减尺度 (m)
HEADING_LOCK_ANGLE = math.radians(25.0)  # 朝向误差超过该阈值时优先旋转
HEADING_LOCK_DECAY = math.radians(60.0)  # 朝向误差增大时速度抑制的衰减范围
TURN_PENALTY_WEIGHT = 2.5  # 朝向未对齐时对高速前进的惩罚权重
MIN_LINEAR_CMD = 0.15  # 最低前进速度指令 (m/s)


# ========================  状态结构体定义  ========================

@dataclasses.dataclass
class RobotState:
    """保存机器人在局部坐标系下的状态信息"""
    x: float
    y: float
    yaw: float
    v: float
    w: float


# ========================  DWA 控制器实现  ========================

class DWAController:
    """基于 DWA 的局部规划控制节点"""

    def __init__(self) -> None:
        rospy.init_node("DWA_CONTROLLER", anonymous=False)
        self._lock = threading.Lock()

        # 通过 ROSData 封装最新消息并处理超时
        self._waypoint = ROSData(timeout=WAYPOINT_TIMEOUT, name="waypoint")
        self._scan = ROSData(timeout=LIDAR_TIMEOUT, name="scan")
        self._odom = ROSData(timeout=ODOM_TIMEOUT, name="odom")
        self._goal_reached = False

        # 订阅上层 waypoint / goal 标志 / 激光雷达 / 里程计
        rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self._goal_cb, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=1)

        # 发布底盘速度
        self._cmd_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
        self._rate = rospy.Rate(int(1.0 / DT))

    # --------------------  回调函数  --------------------

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        """缓存最新 waypoint（期望在机器人坐标系下的局部目标）"""
        self._waypoint.set(msg.data)

    def _goal_cb(self, msg: Bool) -> None:
        """监听上层导航是否宣告到达终点"""
        self._goal_reached = msg.data

    def _scan_cb(self, msg: LaserScan) -> None:
        """缓存激光雷达扫描"""
        self._scan.set(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """缓存里程计信息"""
        self._odom.set(msg)

    # --------------------  主循环  --------------------

    def run(self) -> None:
        """主循环：等待数据 -> 求解最佳速度 -> 发布 /cmd_vel"""
        rospy.loginfo("DWA controller is ready.")
        while not rospy.is_shutdown():
            twist = Twist()

            # 若全局导航已完成，保持停止
            if self._goal_reached:
                self._cmd_pub.publish(twist)
                rospy.loginfo_throttle(1.0, "Goal reached flag set. Holding still.")
                self._rate.sleep()
                continue

            # 数据不全时等待
            if not self._data_ready():
                self._cmd_pub.publish(twist)
                self._rate.sleep()
                continue

            # 提取当前状态，并仅使用 waypoint 的前两个分量 (dx, dy)
            state = self._get_state()
            waypoint = np.array(self._waypoint.get(), dtype=np.float32)
            waypoint = waypoint[:2] if waypoint.shape[0] >= 2 else waypoint

            # 调用 DWA 搜索最佳线速度/角速度
            best_v, best_w = self._plan(state, waypoint, self._scan.get())
            if best_v > 0.0 and best_v < MIN_LINEAR_CMD:
                best_v = MIN_LINEAR_CMD
            twist.linear.x = best_v
            twist.angular.z = best_w

            rospy.loginfo("发布速度 -> v: %.3f m/s, w: %.3f rad/s", best_v, best_w)
            self._cmd_pub.publish(twist)
            self._rate.sleep()

    # --------------------  数据检查与状态转换  --------------------

    def _data_ready(self) -> bool:
        """确保 waypoint / scan / odom 均在超时范围内"""
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
        """从里程计提取机器人当前状态（在世界坐标系）"""
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

    # --------------------  DWA 核心搜索  --------------------

    def _plan(self, state: RobotState, waypoint: np.ndarray, scan: LaserScan) -> Tuple[float, float]:
        """基于 DWA 的速度采样与轨迹评估"""
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

                # 障碍余量硬约束：不足直接剔除
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
            rospy.logwarn_throttle(1.0, "未找到可行的速度候选，保持原地。")
        else:
            rospy.logdebug("最佳评分 %.3f (v=%.3f, w=%.3f)", best_score, best_cmd[0], best_cmd[1])

        return best_cmd

    # --------------------  工具函数  --------------------

    def _compute_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """结合速度上限和加速度约束，构造当前时刻的速度可行域"""
        v_min = max(0.0, state.v - ACC_V * DT)
        v_max = min(MAX_V, state.v + ACC_V * DT)
        w_min = max(-MAX_W, state.w - ACC_W * DT)
        w_max = min(MAX_W, state.w + ACC_W * DT)
        return v_min, v_max, w_min, w_max

    def _simulate(self, state: RobotState, v: float, w: float) -> List[RobotState]:
        """在机器人局部坐标系内积分预测轨迹"""
        traj: List[RobotState] = []
        x, y, yaw = 0.0, 0.0, 0.0  # 以机器人当前姿态为原点

        for _ in range(PRED_HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            traj.append(RobotState(x, y, yaw, v, w))

        return traj

    def _heading_error(self, state: RobotState, waypoint: np.ndarray) -> float:
        """返回末端姿态相对 waypoint 方向的偏差 (rad)"""
        if waypoint.shape[0] < 2:
            return 0.0
        goal_heading = math.atan2(waypoint[1], waypoint[0])
        return clip_angle(goal_heading - state.yaw)

    def _heading_cost(self, state: RobotState, waypoint: np.ndarray) -> float:
        """末端朝向越接近 waypoint 方向，得分越高 (0~1)"""
        angle_diff = abs(self._heading_error(state, waypoint))
        return 1.0 - angle_diff / math.pi

    def _goal_distance_cost(self, state: RobotState, waypoint: np.ndarray) -> float:
        """终点距离 waypoint 越近，得分越高 (0~1)"""
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
        估算轨迹下的最小障碍余量，考虑机器人半径并进行归一化。
        返回 0~1 的得分，1 表示远离障碍，0 表示紧贴障碍。
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

                # 激光距离减去车体半径，相当于对障碍物膨胀
                obstacle_dist = max(obstacle_dist - ROBOT_RADIUS, range_min)

                if obstacle_dist > range_min:
                    margin = max(obstacle_dist - dist, 0.0)
                    min_clearance = min(min_clearance, margin)

        min_clearance = max(min_clearance, 0.0)
        denominator = max(effective_max, 1e-3)
        return min(min_clearance / denominator, 1.0)

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """四元数转 yaw（Z 轴朝向）"""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


# ========================  节点入口  ========================

def main() -> None:
    controller = DWAController()
    controller.run()


if __name__ == "__main__":
    main()










