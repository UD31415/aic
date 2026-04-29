#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
WaveAndCollect: wave the arm back and forth while saving left/center/right
camera images to disk at each observation step.

Output layout:
    <DATA_ROOT>/
      <scene_name>/
        point_000_left.png
        point_000_center.png
        point_000_right.png
        point_001_left.png
        ...
        trajectory.json      ← waypoint poses + per-image filenames

Scene name is built from Task fields + a timestamp so each run is unique.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration

class WaveAndCollect(Policy):
    """
    Waves the arm along a sinusoidal Y-axis path (same motion as WaveArm)
    and captures left / center / right images at each observation step,
    writing them to disk grouped by scene.

    Tunable class-level constants:
      WAVE_X           fixed X position during the wave (meters)
      WAVE_Y_CENTER    center of the Y sweep (meters)
      WAVE_Y_HALF_AMP  half-amplitude of the Y sweep (meters)
      WAVE_Z           fixed Z height during the wave (meters)
      LOOP_SECONDS     period of one full Y oscillation (seconds)
      TIMEOUT_S        total duration of the wave-and-collect run (seconds)
      STEP_SLEEP_S     sleep between observation steps (seconds)
      DATA_ROOT        root directory for saved data (relative or absolute)
    """

    WAVE_X: float = -0.4
    WAVE_Y_CENTER: float = 0.45
    WAVE_Y_HALF_AMP: float = 0.3
    WAVE_Z: float = 0.25
    LOOP_SECONDS: float = 5.0
    TIMEOUT_S: float = 10.0
    STEP_SLEEP_S: float = 0.25

    DATA_ROOT: str = "/home/administrato/dev/ws_aic/data"

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.get_logger().info("WaveAndCollect: initialized.")

    @staticmethod
    def _save_image(image_msg, path: Path) -> None:
        """Write a sensor_msgs/Image (RGB8) to a PNG file, converting RGB→BGR."""
        rgb = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info("WaveAndCollect.insert_cable() start")

        # Build a unique scene directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_parts = [
            p for p in [task.cable_name, task.plug_name, task.port_name, timestamp]
            if p
        ]
        scene_name = "__".join(scene_parts) if scene_parts else timestamp
        scene_dir = Path(self.DATA_ROOT) / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"WaveAndCollect: saving scene to '{scene_dir}'")



        start_time = self.time_now()
        timeout = Duration(seconds=self.TIMEOUT_S)
        idx = 0
        send_feedback("waving arm and collecting images")

        while (self.time_now() - start_time) < timeout:
            self.sleep_for(self.STEP_SLEEP_S)

            observation = get_observation()
            if observation is None:
                self.get_logger().info("No observation received.")
                continue

            t = (
                observation.center_image.header.stamp.sec
                + observation.center_image.header.stamp.nanosec / 1e9
            )

            # Compute sinusoidal Y position from observation timestamp
            loop_fraction = (t % self.LOOP_SECONDS) / self.LOOP_SECONDS
            y_scale = 2.0 * loop_fraction
            if y_scale > 1.0:
                y_scale = 2.0 - y_scale
            y_scale -= 1.0  # oscillates linearly between [-1, 1]

            pose = Pose(
                position=Point(
                    x=self.WAVE_X,
                    y=self.WAVE_Y_CENTER + self.WAVE_Y_HALF_AMP * y_scale,
                    z=self.WAVE_Z,
                ),
                orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
            )

            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.get_logger().info(
                f"  step {idx:4d} | t={t:.2f}s | y={pose.position.y:.4f}"
            )

            # Save left / center / right images
            prefix = f"point_{idx:03d}"
            left_name   = f"{prefix}_left.png"
            center_name = f"{prefix}_center.png"
            right_name  = f"{prefix}_right.png"

            self._save_image(observation.left_image,   scene_dir / left_name)
            self._save_image(observation.center_image, scene_dir / center_name)
            self._save_image(observation.right_image,  scene_dir / right_name)

            idx += 1


        self.get_logger().info(
            f"WaveAndCollect: done — {idx} steps captured, metadata "
        )

        return True
