#!/usr/bin/env python3
"""
Record waypoint commands from /waypoint (std_msgs/Float32MultiArray) and export CSV + PNG
without drawing heading arrows.

Run:
    python waypoint_rec.py
    # 可选参数:
    # python waypoint_rec.py --topic /my_waypoint_topic --log-dir ~/fk/alt_dir
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float32MultiArray


@dataclass
class PoseSample:
    stamp: float
    x: float
    y: float
    yaw: Optional[float] = None  # yaw 仍可保存到 CSV，但不用于绘图


class WaypointLogger:
    """Collects waypoint poses and exports CSV + PNG."""

    def __init__(
        self,
        log_dir: str | Path,
        csv_name: str = "waypoint_trajectory.csv",
        image_name: str = "waypoint_trajectory.png",
    ) -> None:
        self._samples: List[PoseSample] = []
        self._log_dir = Path(log_dir).expanduser()
        self._csv_path = self._log_dir / csv_name
        self._image_path = self._log_dir / image_name
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def samples(self) -> Sequence[PoseSample]:
        return tuple(self._samples)

    def record(self, x: float, y: float, yaw: Optional[float], stamp: float) -> None:
        self._samples.append(
            PoseSample(
                stamp=stamp,
                x=float(x),
                y=float(y),
                yaw=None if yaw is None else float(yaw),
            )
        )

    def save_csv(self) -> Path:
        if not self._samples:
            raise RuntimeError("No waypoint samples captured yet.")

        with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestamp", "dx", "dy", "yaw(rad)"])
            for sample in self._samples:
                writer.writerow(
                    [
                        f"{sample.stamp:.6f}",
                        f"{sample.x:.6f}",
                        f"{sample.y:.6f}",
                        "" if sample.yaw is None else f"{sample.yaw:.6f}",
                    ]
                )
        return self._csv_path

    def plot(
        self,
        title: str = "Waypoint Trajectory",
        dpi: int = 150,
    ) -> Path:
        if not self._samples:
            raise RuntimeError("No waypoint samples captured yet.")

        xs = [sample.x for sample in self._samples]
        ys = [sample.y for sample in self._samples]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.plot(xs, ys, linewidth=2, linestyle="--", color="#ff7f0e", label="Waypoints")
        ax.scatter(xs, ys, color="#ff7f0e", s=40)
        ax.scatter(xs[0], ys[0], color="green", marker="o", s=80, label="Start")
        ax.scatter(xs[-1], ys[-1], color="red", marker="x", s=80, label="Last")

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("dx (m)")
        ax.set_ylabel("dy (m)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._image_path)
        plt.close(fig)

        return self._image_path


class WaypointRecorderNode:
    """ROS node that subscribes to Float32MultiArray waypoints and logs them."""

    def __init__(self, topic: str, log_dir: str) -> None:
        self._logger = WaypointLogger(log_dir=log_dir)
        self._topic = topic

        rospy.Subscriber(self._topic, Float32MultiArray, self._cb, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("Recording waypoints from %s", self._topic)

    def _cb(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 2:
            rospy.logwarn_throttle(5.0, "Waypoint message shorter than 2 elements; skipping.")
            return

        dx, dy = data[0], data[1]

        yaw = None
        if len(data) >= 4:
            hx, hy = data[2], data[3]
            if not math.isclose(hx, 0.0, abs_tol=1e-6) or not math.isclose(hy, 0.0, abs_tol=1e-6):
                yaw = math.atan2(hy, hx)

        stamp = rospy.get_time()
        if stamp == 0.0:
            stamp = time.time()

        self._logger.record(x=dx, y=dy, yaw=yaw, stamp=stamp)

    def _on_shutdown(self) -> None:
        if not self._logger.samples:
            rospy.logwarn("No waypoint samples received; nothing to save.")
            return
        try:
            csv_path = self._logger.save_csv()
            img_path = self._logger.plot()
            rospy.loginfo("Saved waypoint CSV to %s", csv_path)
            rospy.loginfo("Saved waypoint plot to %s", img_path)
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logerr("Failed to export waypoint trajectory: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record waypoint trajectory from ROS Float32MultiArray topic.")
    parser.add_argument(
        "--topic",
        default="/waypoint",
        help="Waypoint topic (std_msgs/Float32MultiArray) (default: /waypoint)",
    )
    parser.add_argument(
        "--log-dir",
        default="~/fk/trajectory",
        help="Directory for CSV/PNG output (default: ~/fk/ME5413_Final_Project/visualnav-transformer/deployment/trajectory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("waypoint_trajectory_logger", anonymous=True)
    WaypointRecorderNode(topic=args.topic, log_dir=args.log_dir)
    rospy.spin()


if __name__ == "__main__":
    main()
