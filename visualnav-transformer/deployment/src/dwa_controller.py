#!/usr/bin/env python3
"""
dwa_controller.py -- Standardized DWA integrated as a ROS node.

Features:
- Uses odometry (/odom by default) and LaserScan ('/front/scan' by your note) to build state.
- Builds dynamic window from Vm (limits), Vd (accel), Va (obstacle distance).
- Predicts candidate trajectories and scores them by heading/clearance/velocity.
- Publishes Twist to vel_navi_topic from config.
- Falls back to PD-style controller if odom/scan missing.
"""

import math
import time
from typing import Tuple, List

import numpy as np
import yaml
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from topic_names import (WAYPOINT_TOPIC, REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle

# -------------------------
# Load robot config
# -------------------------
CONFIG_PATH = "../config/robot.yaml"
DEFAULTS = {
    "max_v": 0.2,
    "max_w": 0.2,
    "max_acc_v": 0.5,
    "max_acc_w": 0.5,
    "robot_radius": 0.25,
    "laser_topic": "/front/scan",
    "odom_topic": "/odometry/filtered",
    "vel_navi_topic": "/cmd_vel",
    "frame_rate": 10,
    # DWA params
    "predict_time": 0.2,
    "dt_predict": 0.1,
    "v_sample": 0.05,
    "w_sample": 0.1,
    "alpha": 1.0,   # heading weight
    "beta": 0.8,    # clearance weight
    "gamma": 0.5,   # velocity weight
    "judge_distance": 1.0,  # large constant when no nearby obstacles
    "goal_threshold": 0.01
}

try:
    with open(CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f) or {}
except Exception as e:
    rospy.logwarn(f"Could not read {CONFIG_PATH}: {e}")
    robot_config = {}
cfg = DEFAULTS.copy()
cfg.update(robot_config)

# core params
MAX_V = float(cfg["max_v"])
MAX_W = float(cfg["max_w"])
MAX_ACC_V = float(cfg["max_acc_v"])
MAX_ACC_W = float(cfg["max_acc_w"])
ROBOT_RADIUS = float(cfg["robot_radius"])
LASER_TOPIC = str(cfg["laser_topic"])
ODOM_TOPIC = str(cfg["odom_topic"])
VEL_TOPIC = str(cfg["vel_navi_topic"])
FRAME_RATE = float(cfg["frame_rate"])
DT = 1.0 / FRAME_RATE

# DWA params
PREDICT_TIME = float(cfg["predict_time"])
DT_PRED = float(cfg["dt_predict"])
V_SAMPLE = float(cfg["v_sample"])
W_SAMPLE = float(cfg["w_sample"])
ALPHA = float(cfg["alpha"])
BETA = float(cfg["beta"])
GAMMA = float(cfg["gamma"])
JUDGE_DISTANCE = float(cfg["judge_distance"])
GOAL_THRESHOLD = float(cfg["goal_threshold"])

# runtime
WAYPOINT_TIMEOUT = 1.0
RATE = int(max(5, round(FRAME_RATE / 2)))
EPS = 1e-8

# globals
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
current_odom_msg = None
current_scan_msg = None

# -------------------------
# Utilities
# -------------------------
def scan_to_points(scan: LaserScan) -> np.ndarray:
    """Convert LaserScan to Nx2 numpy array in robot frame."""
    if scan is None:
        return np.empty((0, 2))
    ranges = np.array(scan.ranges)
    # create angles array consistent with ranges length
    n = len(ranges)
    angles = scan.angle_min + np.arange(n) * scan.angle_increment
    mask = np.isfinite(ranges) & (ranges > 0.01) & (ranges < (scan.range_max if scan.range_max > 0 else 100.0))
    if not np.any(mask):
        return np.empty((0, 2))
    rs = ranges[mask]
    angs = angles[mask]
    xs = rs * np.cos(angs)
    ys = rs * np.sin(angs)
    return np.stack([xs, ys], axis=1)


def odom_to_state(odom: Odometry) -> List[float]:
    """Return [x, y, yaw, v, w] in robot base frame (x,y are odom frame)"""
    if odom is None:
        return None
    px = odom.pose.pose.position.x
    py = odom.pose.pose.position.y
    # yaw from quaternion
    q = odom.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    vx = odom.twist.twist.linear.x
    wz = odom.twist.twist.angular.z
    return [px, py, yaw, vx, wz]


# -------------------------
# Callbacks
# -------------------------
def callback_waypoint(msg: Float32MultiArray):
    waypoint.set(msg.data)


def callback_reached(msg: Bool):
    global reached_goal
    reached_goal = bool(msg.data)


def callback_odom(msg: Odometry):
    global current_odom_msg
    current_odom_msg = msg


def callback_scan(msg: LaserScan):
    global current_scan_msg
    current_scan_msg = msg


# -------------------------
# PD fallback (safe)
# -------------------------
def pd_fallback(wp: list) -> Tuple[float, float]:
    """Simple PD-like fallback used if odom/scan are missing or DWA fails."""
    if wp is None:
        return 0.0, 0.0
    wp = np.array(wp)
    if len(wp) == 2:
        dx, dy = float(wp[0]), float(wp[1])
    else:
        dx, dy, hx, hy = float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])
    if len(wp) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0.0
        w = clip_angle(math.atan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * math.pi / (2 * DT)
    else:
        v = dx / DT
        w = math.atan2(dy, dx) / DT
    v = float(np.clip(v, -MAX_V, MAX_V))
    w = float(np.clip(w, -MAX_W, MAX_W))
    return v, w


# -------------------------
# DWA core: dynamic window
# -------------------------
def calc_v_limits() -> Tuple[float, float, float, float]:
    """Vm: system velocity limits [v_min, v_max, w_min, w_max]"""
    return 0.0, MAX_V, -MAX_W, MAX_W  # Disable reverse by default


def calc_accel_window(v_cur: float, w_cur: float) -> Tuple[float, float, float, float]:
    """Vd: acceleration-limited window around current velocities"""
    v_low = v_cur - MAX_ACC_V * DT
    v_high = v_cur + MAX_ACC_V * DT
    w_low = w_cur - MAX_ACC_W * DT
    w_high = w_cur + MAX_ACC_W * DT
    return v_low, v_high, w_low, w_high


def calc_obstacle_window(state: List[float], obstacles: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Va: obstacle-limited window. Use distance to nearest obstacle to bound safe max velocity.
    Improved version: Only consider obstacles in front of robot (within +/- 45 degrees) and add minimum velocity.
    """
    if state is None:
        return 0.0, MAX_V, -MAX_W, MAX_W  # Disable reverse by default
    # compute distance from robot position to nearest obstacle
    if obstacles is None or obstacles.size == 0:
        return 0.0, MAX_V, -MAX_W, MAX_W  # Disable reverse by default
    
    # 过滤只考虑机器人前方45度范围内的障碍物
    # 计算每个障碍物点的角度
    angles = np.arctan2(obstacles[:, 1], obstacles[:, 0])
    # 只保留前方45度范围内的障碍物（-pi/4到pi/4）
    front_mask = np.abs(angles) <= np.pi / 4
    front_obstacles = obstacles[front_mask]
    
    # 如果前方没有障碍物，允许正常速度
    if front_obstacles.size == 0:
        return 0.0, MAX_V, -MAX_W, MAX_W
    
    # 计算到前方障碍物的距离
    dists = np.hypot(front_obstacles[:, 0], front_obstacles[:, 1])
    min_d = float(np.min(dists)) if dists.size > 0 else float('inf')
    
    # 改进的安全速度计算
    safe_margin = max(0.0, min_d - ROBOT_RADIUS)
    
    # 添加最小速度以避免完全停止，同时根据距离适当调整
    if safe_margin <= 0.1:  # 非常近的障碍物
        v_high = 0.05  # 非常慢的速度
    elif safe_margin <= 0.3:  # 较近的障碍物
        v_high = 0.1  # 较慢的速度
    else:
        # 使用制动距离公式，但增加最小速度
        v_calculated = math.sqrt(2.0 * MAX_ACC_V * safe_margin)
        v_high = max(0.1, v_calculated)  # 确保有最小速度
    
    # Disable reverse by default to prevent backward motion
    v_low = 0.0  # Changed from -MAX_V to 0.0
    # keep angular window as system limits
    w_low, w_high = -MAX_W, MAX_W
    # clip
    v_high = min(v_high, MAX_V)
    return v_low, v_high, w_low, w_high


# -------------------------
# Trajectory prediction & evaluation
# -------------------------
def predict_trajectory_from_state(state: List[float], v: float, w: float) -> np.ndarray:
    """
    Simulate trajectory in robot-local coordinates.
    We will translate state to robot-local frame where robot starts at (0,0,0) for easier collision checking.
    state: [x, y, yaw, v_cur, w_cur] in odom frame -- but prediction done in robot frame -> use relative goal earlier.
    Return Nx3 array of (x,y,yaw) in robot-local coordinates.
    """
    # start at origin in robot frame
    x, y, yaw = 0.0, 0.0, 0.0
    traj = []
    t = 0.0
    while t <= PREDICT_TIME + 1e-9:
        traj.append((x, y, yaw))
        # integrate
        x += v * math.cos(yaw) * DT_PRED
        y += v * math.sin(yaw) * DT_PRED
        yaw = clip_angle(yaw + w * DT_PRED)
        t += DT_PRED
    return np.array(traj)


def heading_cost(traj: np.ndarray, goal_dx: float, goal_dy: float) -> float:
    """Compute heading score: smaller angle between final pose and goal => higher score."""
    fx, fy, fyaw = traj[-1]
    # vector from final pose to goal in robot frame
    dx = goal_dx - fx
    dy = goal_dy - fy
    target_angle = math.atan2(dy, dx)
    angle_diff = abs(clip_angle(target_angle - fyaw))
    # convert to a score where smaller angle yields larger value
    # use (pi - angle_diff) so that heading aligned -> high
    return (math.pi - angle_diff)


def clearance_cost(traj: np.ndarray, obs_pts: np.ndarray) -> float:
    """
    Compute minimum clearance along trajectory minus robot radius.
    Improved version with weighted obstacle importance based on direction.
    """
    if obs_pts is None or obs_pts.size == 0:
        return JUDGE_DISTANCE  # very large "clearance"
    rr = ROBOT_RADIUS
    min_dist = float('inf')
    
    # 为轨迹上的每个点计算安全距离
    for i, p in enumerate(traj):
        # 计算每个障碍物点到轨迹点的距离
        dists = np.hypot(obs_pts[:, 0] - p[0], obs_pts[:, 1] - p[1])
        
        # 计算障碍物相对于轨迹点的角度
        angles = np.arctan2(obs_pts[:, 1] - p[1], obs_pts[:, 0] - p[0])
        
        # 根据角度分配权重 - 前方障碍物权重更高
        # 前方45度内权重为1.0，侧边权重递减，后方权重最小
        weights = np.ones_like(angles)
        front_mask = np.abs(angles) <= np.pi / 4  # 前方45度
        side_mask = np.abs(angles) > np.pi / 4  # 侧面和后方
        
        # 侧面障碍物权重降低
        weights[side_mask] = 0.5
        
        # 考虑时间因素 - 越近的未来点越重要
        time_weight = 1.0 - (i / len(traj)) * 0.5
        
        # 应用权重调整距离
        weighted_dists = dists * weights * time_weight
        
        md = np.min(weighted_dists) if weighted_dists.size > 0 else float('inf')
        if md < min_dist:
            min_dist = md
        if min_dist <= rr * 0.8:  # 稍微放宽碰撞检测阈值
            # 接近碰撞
            return -1.0
    
    # clearance measured as min_dist - rr
    return max(0.0, min_dist - rr)


def velocity_cost(v: float) -> float:
    """Prefer forward speed and penalize backward motion."""
    if v < 0:  # Penalize backward motion
        return 0.0  # Give very low cost to backward motion
    return v  # Return actual forward speed as cost (higher is better)


# -------------------------
# DWA planner wrapper
# -------------------------
def dwa_plan(goal_dx: float, goal_dy: float, odom_msg: Odometry, scan_msg: LaserScan) -> Tuple[float, float]:
    """
    Return (v, w).
    goal_dx/goal_dy are in robot base frame (relative).
    odom_msg and scan_msg may be None -> fallback to PD.
    """
    # fallback if no odom
    if odom_msg is None:
        rospy.logdebug("No odom available, using PD fallback.")
        return pd_fallback([goal_dx, goal_dy])

    # construct state and obstacles in robot frame
    state = odom_to_state(odom_msg)  # [x,y,yaw,v_cur,w_cur] in odom frame
    # convert scan to points in robot frame (already in robot frame)
    obs_pts = scan_to_points(scan_msg) if scan_msg is not None else np.empty((0, 2))

    # current velocities (from odom)
    v_cur = state[3]
    w_cur = state[4]

    # 1) build dynamic window: intersection of Vm, Vd, Va
    Vm = calc_v_limits()
    Vd = calc_accel_window(v_cur, w_cur)
    Va = calc_obstacle_window(state, obs_pts)
    v_min = max(Vm[0], Vd[0], Va[0])
    v_max = min(Vm[1], Vd[1], Va[1])
    w_min = max(Vm[2], Vd[2], Va[2])
    w_max = min(Vm[3], Vd[3], Va[3])

    # ensure non-empty window
    if v_min > v_max:
        v_min, v_max = Vm[0], Vm[1]
    if w_min > w_max:
        w_min, w_max = Vm[2], Vm[3]

    best_score = -float('inf')
    best_v, best_w = 0.0, 0.0

    # sample velocities (including backwards if allowed)
    v_range = np.arange(v_min, v_max + 1e-9, V_SAMPLE)
    # if v_range is empty, add zero
    if v_range.size == 0:
        v_range = np.array([0.0])
    w_range = np.arange(w_min, w_max + 1e-9, W_SAMPLE)
    if w_range.size == 0:
        w_range = np.array([0.0])

    for v in v_range:
        for w in w_range:
            # predict in robot frame starting at origin
            traj = predict_trajectory_from_state(state, v, w)
            # clearance (if negative -> collision -> skip)
            clear = clearance_cost(traj, obs_pts)
            if clear < 0:
                continue
            # heading and velocity score
            h = heading_cost(traj, goal_dx, goal_dy)
            vel_s = velocity_cost(v)
            # combine using weights, higher is better
            score = ALPHA * h + BETA * clear + GAMMA * vel_s
            if score > best_score:
                best_score = score
                best_v, best_w = v, w

    # if nothing valid found, try in-place rotation escape
    if best_score == -float('inf'):
        # try rotate left or right
        for candidate_w in (MAX_W * 0.5, -MAX_W * 0.5):
            traj = predict_trajectory_from_state(state, 0.0, candidate_w)
            clear = clearance_cost(traj, obs_pts)
            if clear >= 0:
                return 0.0, float(np.clip(candidate_w, -MAX_W, MAX_W))
        # else stop
        return 0.0, 0.0

    # clip final
    return float(np.clip(best_v, -MAX_V, MAX_V)), float(np.clip(best_w, -MAX_W, MAX_W))


# -------------------------
# Main node
# -------------------------
def main():
    global reached_goal
    rospy.init_node("dwa_controller_node", anonymous=False)

    rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_waypoint, queue_size=1)
    rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached, queue_size=1)
    rospy.Subscriber(ODOM_TOPIC, Odometry, callback_odom, queue_size=1)
    rospy.Subscriber(LASER_TOPIC, LaserScan, callback_scan, queue_size=1)

    vel_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
    rate = rospy.Rate(RATE)

    rospy.loginfo(f"DWA controller started. Laser: {LASER_TOPIC}, Odom: {ODOM_TOPIC}, Vel out: {VEL_TOPIC}")

    while not rospy.is_shutdown():
        twist = Twist()
        if reached_goal:
            vel_pub.publish(twist)
            rospy.loginfo("Reached goal -> stopping dwa_controller.")
            return

        if waypoint.is_valid(verbose=False):
            wp = waypoint.get()
            if len(wp) >= 2:
                goal_dx = float(wp[0])
                goal_dy = float(wp[1])
            else:
                goal_dx, goal_dy = 0.0, 0.0

            dist = math.hypot(goal_dx, goal_dy)
            if dist < GOAL_THRESHOLD:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                v_cmd, w_cmd = dwa_plan(goal_dx, goal_dy, current_odom_msg, current_scan_msg)
                twist.linear.x = float(np.clip(v_cmd, -MAX_V, MAX_V))
                twist.angular.z = float(np.clip(w_cmd, -MAX_W, MAX_W))
                # 打印规划的速度信息
                rospy.loginfo(f"v={twist.linear.x:.3f} m/s, w={twist.angular.z:.3f} rad/s")
        else:
            # no waypoint -> stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # 打印停止信息
            rospy.loginfo("stop")

        vel_pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
