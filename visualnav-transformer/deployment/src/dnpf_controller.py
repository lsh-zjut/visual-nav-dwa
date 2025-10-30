#!/usr/bin/env python3
import os, sys, importlib.util, types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DNPF_PATH = os.path.join(SCRIPT_DIR, "utils_dnpf")

# ✅ 映射 utils_dnpf → utils
if os.path.isdir(UTILS_DNPF_PATH):
    sys.path.append(UTILS_DNPF_PATH)
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [UTILS_DNPF_PATH]
    sys.modules["utils"] = utils_pkg
    sys.path.append(SCRIPT_DIR)
else:
    raise FileNotFoundError(f"❌ utils_dnpf not found at {UTILS_DNPF_PATH}")

import rospy
import torch
import numpy as np
import yaml
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool

# ============================================================
#                   路径修正（关键部分）
# ============================================================

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(SCRIPT_DIR)
# sys.path.append(os.path.join(SCRIPT_DIR, "utils_dnpf"))  # ✅ DNPF专用utils目录

from model_nn import Autoencoder_path  # DNPF模型结构

# ============================================================
#                    参数 & 全局变量
# ============================================================

CONFIG_PATH = os.path.join(SCRIPT_DIR, "../config/robot.yaml")
MODEL_PATH = os.path.join(SCRIPT_DIR, "../model_weights/NPField_Dynamic_10_A100.pth")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MAX_V = cfg["max_v"]
MAX_W = cfg["max_w"]
VEL_TOPIC = cfg["vel_teleop_topic"]
SCAN_TOPIC = cfg["base_scan_topic"]
DT = 1.0 / cfg["frame_rate"]

waypoint = None
latest_scan = None
reached_goal = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#                   模型加载
# ============================================================

rospy.loginfo(f"[DNPF Controller] Loading model from {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = Autoencoder_path(mode="k").to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
rospy.loginfo(f"[DNPF Controller] Model loaded successfully ({total_params/1e6:.2f} M parameters).")

# ============================================================
#                   ROS回调函数
# ============================================================

def cb_waypoint(msg):
    global waypoint
    waypoint = np.array(msg.data, dtype=np.float32)
    rospy.loginfo_throttle(2.0, f"[DNPF] New waypoint: {waypoint}")

def cb_scan(msg):
    global latest_scan
    latest_scan = np.array(msg.ranges, dtype=np.float32)

def cb_goal(msg):
    global reached_goal
    reached_goal = msg.data

# ============================================================
#                   工具函数
# ============================================================

def lidar_to_map(scan, grid_size=50, res=0.1, fov=np.pi):
    """
    将激光雷达数据转换为局部50x50占据图
    """
    if scan is None:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    cx, cy = grid_size // 2, grid_size // 2
    angles = np.linspace(-fov/2, fov/2, len(scan))

    for r, a in zip(scan, angles):
        if not np.isfinite(r) or r > 4.9:
            continue
        x = int(cx + (r * np.cos(a)) / res)
        y = int(cy + (r * np.sin(a)) / res)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid[y, x] = 1.0
    return grid


def compute_dnpf_control(model, waypoint, scan):
    """
    使用神经势场预测避障方向并输出平滑控制
    """
    if waypoint is None or scan is None:
        return 0.0, 0.0

    occ_grid = lidar_to_map(scan)
    footprint = np.zeros((50, 50), dtype=np.float32)

    map_flat = occ_grid.flatten()
    foot_flat = footprint.flatten()
    dx, dy = waypoint[0], waypoint[1]
    theta = np.arctan2(dy, dx)
    inp_vec = np.concatenate([map_flat, foot_flat, [0.0, 0.0, theta]])
    inp_tensor = torch.tensor(inp_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        emb, _ = model.encode_map_footprint(inp_tensor)
        test_batch = torch.hstack((
            emb,
            torch.tensor([[dx]], device=device),
            torch.tensor([[dy]], device=device),
            torch.tensor([[theta]], device=device)
        ))
        potential = model.encode_map_pos(test_batch)[0].cpu().numpy()

    # === 计算梯度 ===
    # 确保potential是二维数组
    if potential.ndim > 2:
        # 如果是高维数组，展平到二维
        potential = potential.reshape(-1, potential.shape[-1])
    elif potential.ndim == 1:
        # 如果是一维数组，扩展为二维
        potential = potential.reshape(1, -1)
    
    # 检查数组尺寸是否足够计算梯度（至少每个维度有2个元素）
    if potential.shape[0] < 2 or potential.shape[1] < 2:
        # 如果数组太小，无法计算梯度，使用默认梯度值
        grad_x, grad_y = 0.0, 0.0
    else:
        # 计算梯度
        gradients = np.gradient(potential)
        gy, gx = gradients[:2]  # 只取前两个维度的梯度
        grad_x, grad_y = np.mean(gx), np.mean(gy)

    # === 计算角度与距离误差 ===
    heading_error = np.arctan2(dy, dx)
    distance = np.linalg.norm([dx, dy])

    # === 控制律 ===
    v = np.clip(distance * 0.3, 0.05, MAX_V)
    w = np.clip(heading_error * 1.5, -MAX_W, MAX_W)

    # === 平滑滤波 ===
    alpha = 0.2
    v_prev = getattr(compute_dnpf_control, "_last_v", v)
    w_prev = getattr(compute_dnpf_control, "_last_w", w)
    v = alpha * v + (1 - alpha) * v_prev
    w = alpha * w + (1 - alpha) * w_prev
    compute_dnpf_control._last_v = v
    compute_dnpf_control._last_w = w

    return float(v), float(w)


# ============================================================
#                   主函数
# ============================================================

def main():
    rospy.init_node("DNPF_CONTROLLER", anonymous=False)
    rospy.Subscriber("/waypoint", Float32MultiArray, cb_waypoint)
    rospy.Subscriber(SCAN_TOPIC, LaserScan, cb_scan)
    rospy.Subscriber("/topoplan/reached_goal", Bool, cb_goal)
    vel_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)

    rospy.loginfo("[DNPF Controller] Node started.")
    rate = rospy.Rate(30)
    vel_msg = Twist()

    while not rospy.is_shutdown():
        if reached_goal:
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            vel_pub.publish(vel_msg)
            rospy.loginfo("[DNPF Controller] Goal reached. Stopping.")
            break

        if waypoint is not None and latest_scan is not None:
            v, w = compute_dnpf_control(model, waypoint, latest_scan)
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            vel_pub.publish(vel_msg)
            rospy.loginfo_throttle(1.0, f"[DNPF] v={v:.3f}, w={w:.3f}")

        rate.sleep()

if __name__ == "__main__":
    main()
