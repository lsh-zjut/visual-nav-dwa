import dataclasses
import math
import threading
from typing import Optional, Tuple, List

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

CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_cfg = yaml.safe_load(f)

MAX_V = float(robot_cfg.get("max_v", 0.25))
MAX_W = float(robot_cfg.get("max_w", 0.3))
VEL_TOPIC = robot_cfg.get("vel_teleop_topic", "/cmd_vel")
SCAN_TOPIC = robot_cfg.get("base_scan_topic", "/front/scan")
ODOM_TOPIC = robot_cfg.get("odom_topic", "/odometry/filtered")

WAYPOINT_TIMEOUT = 1.0
LIDAR_TIMEOUT = 0.5
ODOM_TIMEOUT = 0.5

DT = 0.1
PRED_HORIZON = 12
ACC_V = 0.4
ACC_W = 1.5
RES_V = 0.02
RES_W = 0.05

WEIGHT_HEADING = 1.5
WEIGHT_VELOCITY = 1.0
WEIGHT_CLEARANCE = 1.2
CLEARANCE_HARD = 0.05

@dataclasses.dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    w: float

class DWAController:
    def __init__(self):
        rospy.init_node("DWA_CONTROLLER", anonymous=False)
        self._lock = threading.Lock()

        # 存储最新的 waypoint / 激光雷达 / 里程计数据，并自动处理超时
        self._waypoint = ROSData(timeout=WAYPOINT_TIMEOUT, name="waypoint")
        self._goal_reached = False
        self._scan = ROSData(timeout=LIDAR_TIMEOUT, name="scan")
        self._odom = ROSData(timeout=ODOM_TIMEOUT, name="odom")

        # 订阅导航目标、终点标志、激光雷达和里程计
        rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self._goal_cb, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=1)

        # 发布速度控制指令
        self._cmd_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
        self._rate = rospy.Rate(int(1.0 / DT))

    def _waypoint_cb(self, msg: Float32MultiArray):
        # 缓存最新 waypoint
        self._waypoint.set(msg.data)

    def _goal_cb(self, msg: Bool):
        # 接收导航节点是否到达全局目标
        self._goal_reached = msg.data

    def _scan_cb(self, msg: LaserScan):
        # 缓存激光雷达数据
        self._scan.set(msg)

    def _odom_cb(self, msg: Odometry):
        # 缓存里程计数据
        self._odom.set(msg)

    def run(self):
        rospy.loginfo("DWA controller is ready.")
        while not rospy.is_shutdown():
            twist = Twist()

            # 若导航侧标记到达目标，立即停车
            if self._goal_reached:
                self._cmd_pub.publish(twist)
                # rospy.loginfo("发布速度 -> v: 0.000 m/s, w: 0.000 rad/s (目标已完成)")
                rospy.loginfo_throttle(1.0, "Goal reached flag set. Holding still.")
                self._rate.sleep()
                continue

            # 确保 waypoint / scan / odom 都可用
            if not self._data_ready():
                self._cmd_pub.publish(twist)
                # rospy.loginfo("发布速度 -> v: 0.000 m/s, w: 0.000 rad/s (等待传感器数据)")
                self._rate.sleep()
                continue

            state = self._get_state()
            waypoint = np.array(self._waypoint.get(), dtype=np.float32)
            waypoint = waypoint[:2] if waypoint.shape[0] >= 2 else waypoint

            best_v, best_w = self._plan(state, waypoint, self._scan.get())
            twist.linear.x = best_v
            twist.angular.z = best_w

            rospy.loginfo("发布速度 -> v: %.3f m/s, w: %.3f rad/s", best_v, best_w)
            self._cmd_pub.publish(twist)
            self._rate.sleep()

    def _data_ready(self) -> bool:
        # ROSData 自带超时检测，这里判断所有数据是否新鲜
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
        # 从里程计取出位姿和速度，并转成 yaw
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

    def _plan(self, state: RobotState, waypoint: np.ndarray, scan: LaserScan) -> Tuple[float, float]:
        # 核心 DWA：采样速度 + 模拟轨迹 + 评分
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
                traj = self._simulate(state, v, w)  # 预测未来若干步
                heading_cost = self._heading_cost(traj[-1], waypoint)  # 与目标方向的偏差
                velocity_cost = v / MAX_V  # 倾向于更快前进
                clearance_cost = self._clearance_cost(  # 避障评分
                    traj, ranges, angles, scan.range_min, scan.range_max
                )

                # 硬约束：最小安全距离不足则丢弃该速度组合
                if clearance_cost < CLEARANCE_HARD:
                    continue

                score = (
                    WEIGHT_HEADING * heading_cost
                    + WEIGHT_VELOCITY * velocity_cost
                    + WEIGHT_CLEARANCE * clearance_cost
                )

                if score > best_score:
                    best_score = score
                    best_cmd = (v, w)

        if best_score == -float("inf"):
            rospy.logwarn_throttle(1.0, "未找到可行的速度候选，保持原地。")
        else:
            rospy.logdebug(
                "最佳评分: %.3f (v=%.3f, w=%.3f)", best_score, best_cmd[0], best_cmd[1]
            )
        return best_cmd

    def _compute_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        # 根据当前速度和最大加速度生成动态窗口
        v_min = max(0.0, state.v - ACC_V * DT)
        v_max = min(MAX_V, state.v + ACC_V * DT)
        w_min = max(-MAX_W, state.w - ACC_W * DT)
        w_max = min(MAX_W, state.w + ACC_W * DT)
        return v_min, v_max, w_min, w_max

    def _simulate(self, state: RobotState, v: float, w: float) -> List[RobotState]:
        # 在机器人自身坐标系内模拟轨迹（原地约束）
        traj = []
        x, y, yaw = 0.0, 0.0, 0.0
        for _ in range(PRED_HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            traj.append(RobotState(x, y, yaw, v, w))
        return traj

    def _heading_cost(self, state: RobotState, waypoint: np.ndarray) -> float:
        # 目标指向角越接近，得分越高（归一化到 0~1）
        goal_heading = math.atan2(waypoint[1], waypoint[0]) if waypoint.shape[0] >= 2 else 0.0
        angle_diff = clip_angle(goal_heading - state.yaw)
        return 1.0 - abs(angle_diff) / math.pi

    def _clearance_cost(self, traj: List[RobotState], ranges: np.ndarray, angles: np.ndarray,
                        range_min: float, range_max: float) -> float:
        # 遍历模拟轨迹，估算沿道路方向的最小障碍余量
        min_clearance = range_max
        for point in traj:
            px, py = point.x, point.y
            dist = math.hypot(px, py)
            heading = math.atan2(py, px)
            idx = int((heading - angles[0]) / (angles[-1] - angles[0]) * (len(angles) - 1))
            if 0 <= idx < len(ranges):
                obstacle_dist = ranges[idx]
                if not np.isfinite(obstacle_dist):
                    obstacle_dist = range_max
                if obstacle_dist > range_min:
                    min_clearance = min(min_clearance, max(obstacle_dist - dist, 0.0))
        min_clearance = max(min_clearance, 0.0)
        return min(min_clearance / range_max, 1.0)

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        # 四元数 -> yaw 欧拉角
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

def main():
    controller = DWAController()
    controller.run()

if __name__ == "__main__":
    main()
