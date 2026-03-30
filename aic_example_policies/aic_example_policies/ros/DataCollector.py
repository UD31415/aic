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
DataCollector policy: sweep the robot arm over the working area and record
left / center / right camera images at each waypoint.

Trajectory: 10×10 boustrophedon (snake) grid = 100 waypoints per scene.
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
Camera intrinsics are read from the CameraInfo bundled inside each Observation
message (no separate subscription required).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)


class DataCollector(Policy):
    """
    Moves the robot over a rectangular working area on a boustrophedon grid,
    capturing left / center / right images at each of the 100 waypoints and
    writing them to disk, grouped by scene.

    Tunable class-level constants:
      SCAN_X_MIN/MAX   workspace X extents in base_link frame (meters)
      SCAN_Y_MIN/MAX   workspace Y extents in base_link frame (meters)
      SCAN_Z           fixed scan height above the workspace (meters)
      SCAN_GRID_N      points per axis; total waypoints = SCAN_GRID_N² (default 10→100)
      SCAN_ORIENTATION gripper orientation at every scan point as (w, x, y, z)
                       — default 180° about X puts the gripper facing straight down
                       — adjust to match your robot's zero-orientation convention
      SETTLE_TIME_S    seconds to wait after commanding each waypoint before capturing
      DATA_ROOT        root directory for saved data (relative to cwd, or absolute)
      STIFFNESS/DAMPING Cartesian controller gains (see _send_pose_to_gripper)
    """

    # Scanning area in base_link frame (meters) — tune to your workspace
    SCAN_X_MIN: float = 0.30
    SCAN_X_MAX: float = 0.70
    SCAN_Y_MIN: float = -0.20
    SCAN_Y_MAX: float = 0.20
    SCAN_Z: float = 0.40               # fixed height above the table

    # 10×10 = 100 waypoints per scene
    SCAN_GRID_N: int = 10

    # Gripper orientation at scan points (w, x, y, z).
    # (0, 1, 0, 0) = 180° about X → gripper TCP pointing straight down.
    # Adjust if your robot's neutral pose has a different reference orientation.
    SCAN_ORIENTATION: tuple = (0.0, 1.0, 0.0, 0.0)

    # How long to wait for the robot to settle before capturing images
    SETTLE_TIME_S: float = 0.5

    # Root folder for saved images; scene sub-directories are created inside it
    DATA_ROOT: str = "data"

    # Cartesian admittance controller gains (same defaults as Policy.set_pose_target)
    STIFFNESS: list = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]  # N/m and N·m/rad
    DAMPING: list   = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]  # N·s/m and N·m·s/rad

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.get_logger().info("DataCollector: initialized.")

    # ------------------------------------------------------------------
    # Scanning trajectory
    # ------------------------------------------------------------------

    def _make_scan_trajectory(self) -> list[Pose]:
        """
        Build a SCAN_GRID_N × SCAN_GRID_N boustrophedon (snake-row) grid.

        Alternating rows are reversed so the TCP travels row-by-row without
        jumping back to the row start, minimising total path length.

        Returns a flat list of Poses, length = SCAN_GRID_N².
        """
        x_vals = np.linspace(self.SCAN_X_MIN, self.SCAN_X_MAX, self.SCAN_GRID_N)
        y_vals = np.linspace(self.SCAN_Y_MIN, self.SCAN_Y_MAX, self.SCAN_GRID_N)
        w, qx, qy, qz = self.SCAN_ORIENTATION

        poses = []
        for row_idx, x_val in enumerate(x_vals):
            # Reverse the Y direction on odd rows for the snake pattern
            row_y = y_vals if row_idx % 2 == 0 else y_vals[::-1]
            for y_val in row_y:
                poses.append(Pose(
                    position=Point(
                        x=float(x_val),
                        y=float(y_val),
                        z=float(self.SCAN_Z),
                    ),
                    orientation=Quaternion(w=w, x=qx, y=qy, z=qz),
                ))
        return poses

    # ------------------------------------------------------------------
    # Gripper pose command (inlined from Policy.set_pose_target)
    # ------------------------------------------------------------------

    def _send_pose_to_gripper(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
    ) -> None:
        """
        Construct a MotionUpdate and publish it to the gripper.

        Full inline of Policy.set_pose_target so every controller parameter
        (stiffness, damping, wrench gains, trajectory mode) is visible here.

          pose      — target Pose for the gripper TCP in frame_id coordinates
          frame_id  — TF frame that pose is expressed in (default: base_link)

        Controller parameter meanings:
          target_stiffness   6×6 diagonal (flat 36 values): spring stiffness toward
                             the target in x/y/z/rx/ry/rz [N/m, N·m/rad]
          target_damping     6×6 diagonal: velocity damping [N·s/m, N·m·s/rad]
          feedforward_wrench extra force/torque applied at the TCP regardless of error
          wrench_feedback_gains how much measured contact force/torque is fed back:
                             [0.5,0.5,0.5,0,0,0] = 50% force feedback, no torque FB
          trajectory_generation_mode MODE_POSITION → track a position setpoint
                                      MODE_VELOCITY → track a velocity setpoint
        """
        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            # Stiffness and damping are 6×6 diagonal matrices stored as flat 36-element arrays
            target_stiffness=np.diag(self.STIFFNESS).flatten(),
            target_damping=np.diag(self.DAMPING).flatten(),
            # No feedforward force/torque at the TCP
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            # Feed back 50% of measured XYZ contact force; no torque feedback
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            # Track a position target (not a velocity target)
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().warn(f"move_robot exception: {ex}")

    # ------------------------------------------------------------------
    # Image saving
    # ------------------------------------------------------------------

    @staticmethod
    def _save_image(image_msg, path: Path) -> None:
        """
        Write a sensor_msgs/Image (RGB8 encoding) to a PNG file at path.

        Converts RGB → BGR before writing because OpenCV's imwrite expects BGR.
        """
        rgb = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    # ------------------------------------------------------------------
    # Policy entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("DataCollector.insert_cable() start")

        # --- Build a unique scene directory --------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Combine task identifiers so the folder name describes the scene
        scene_parts = [
            p for p in [task.cable_name, task.plug_name, task.port_name, timestamp]
            if p  # skip empty strings
        ]
        scene_name = "__".join(scene_parts) if scene_parts else timestamp
        scene_dir = Path(self.DATA_ROOT) / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"DataCollector: saving scene to '{scene_dir}'")

        # --- Generate scan trajectory ---------------------------------------
        trajectory = self._make_scan_trajectory()
        total = len(trajectory)
        self.get_logger().info(
            f"DataCollector: {total}-point scan "
            f"({self.SCAN_GRID_N}×{self.SCAN_GRID_N} grid, "
            f"z={self.SCAN_Z:.2f} m)"
        )

        # --- Metadata to be written after the scan -------------------------
        metadata = {
            "scene": scene_name,
            "task": {
                "id": task.id,
                "cable_name": task.cable_name,
                "cable_type": task.cable_type,
                "plug_name": task.plug_name,
                "plug_type": task.plug_type,
                "port_name": task.port_name,
                "port_type": task.port_type,
                "target_module_name": task.target_module_name,
            },
            "scan_params": {
                "x_min": self.SCAN_X_MIN,
                "x_max": self.SCAN_X_MAX,
                "y_min": self.SCAN_Y_MIN,
                "y_max": self.SCAN_Y_MAX,
                "z": self.SCAN_Z,
                "grid_n": self.SCAN_GRID_N,
                "orientation_wxyz": list(self.SCAN_ORIENTATION),
                "settle_time_s": self.SETTLE_TIME_S,
            },
            "waypoints": [],
        }

        # --- Main scan loop ------------------------------------------------
        for idx, pose in enumerate(trajectory):
            send_feedback(f"scanning {idx + 1}/{total}")
            self.get_logger().info(
                f"  [{idx + 1:3d}/{total}] → "
                f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
            )

            # Command the gripper to this waypoint
            self._send_pose_to_gripper(move_robot=move_robot, pose=pose)

            # Wait for the robot to settle at the target position
            self.sleep_for(self.SETTLE_TIME_S)

            # Fetch the latest observation; retry briefly if not yet available
            obs = None
            for _ in range(10):
                obs = get_observation()
                if obs is not None:
                    break
                self.sleep_for(0.05)

            if obs is None:
                self.get_logger().warn(
                    f"  Waypoint {idx}: no observation received, skipping images."
                )
                continue

            # Save left / center / right images
            prefix = f"point_{idx:03d}"
            left_name   = f"{prefix}_left.png"
            center_name = f"{prefix}_center.png"
            right_name  = f"{prefix}_right.png"

            self._save_image(obs.left_image,   scene_dir / left_name)
            self._save_image(obs.center_image, scene_dir / center_name)
            self._save_image(obs.right_image,  scene_dir / right_name)

            # Record waypoint in metadata
            metadata["waypoints"].append({
                "index": idx,
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
                "orientation_wxyz": [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ],
                "images": {
                    "left":   left_name,
                    "center": center_name,
                    "right":  right_name,
                },
            })

        # --- Write trajectory metadata -------------------------------------
        meta_path = scene_dir / "trajectory.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.get_logger().info(
            f"DataCollector: scan complete — "
            f"{len(metadata['waypoints'])}/{total} images captured, "
            f"metadata → {meta_path}"
        )

        return True
