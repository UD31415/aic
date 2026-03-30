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
ArucoGripper policy: detect an ArUco marker in the center camera and
move the gripper to the detected pose.

Combines:
  - Image decoding from RunACT (raw bytes → numpy → cv2)
  - Pose commanding from CheatCode / Policy.set_pose_target
"""

import time

import cv2
import cv2.aruco as aruco
import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformException
from transforms3d.quaternions import mat2quat, qmult, quat2mat

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


class ArucoGripper(Policy):
    """
    Policy that detects an ArUco marker in the center camera, transforms its
    pose to the robot base frame, and commands the gripper to move to it.

    Tunable class-level constants — override in a subclass or change here:
      MARKER_LENGTH_M   physical side length of the ArUco marker (meters)
      CAMERA_FRAME      TF2 frame name for the center camera optical frame
      APPROACH_Z_OFFSET distance to hover above the marker along world-Z (meters)
      ARUCO_DICT        ArUco dictionary used when printing/detecting markers
      TASK_DURATION_S   how long to track before declaring success
      CONTROL_RATE_HZ   pose-command rate
    """

    MARKER_LENGTH_M: float = 0.05
    CAMERA_FRAME: str = "center_camera_optical_frame"
    APPROACH_Z_OFFSET: float = 0.10
    ARUCO_DICT: int = aruco.DICT_4X4_50
    TASK_DURATION_S: float = 30.0
    CONTROL_RATE_HZ: float = 10.0

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

        # Latch center-camera intrinsics from the first CameraInfo message
        self._cam_info_sub = parent_node.create_subscription(
            CameraInfo,
            "/center_camera/camera_info",
            self._on_camera_info,
            1,
        )

        # Build ArUco detector (OpenCV >= 4.7 API)
        _dict = aruco.getPredefinedDictionary(self.ARUCO_DICT)
        _params = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(_dict, _params)

        self.get_logger().info(
            "ArucoGripper: initialized, waiting for camera intrinsics..."
        )

    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------

    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Latch intrinsics on the first message; ignore subsequent ones."""
        if self._camera_matrix is None:
            self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self._dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info("ArucoGripper: camera intrinsics latched.")

    # ------------------------------------------------------------------
    # Image conversion (same raw-bytes → numpy pattern as RunACT)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_msg_to_gray(image_msg) -> np.ndarray:
        """Convert a sensor_msgs/Image (RGB8 encoding) to a grayscale array."""
        rgb = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height, image_msg.width, 3
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ------------------------------------------------------------------
    # ArUco detection → Pose in camera frame
    # ------------------------------------------------------------------

    def _detect_marker_pose_in_camera(self, image_msg) -> Pose | None:
        """
        Detect ArUco markers in image_msg.

        Returns the first detected marker's Pose expressed in the camera
        optical frame, or None if no marker is visible.
        """
        if self._camera_matrix is None:
            return None

        gray = self._image_msg_to_gray(image_msg)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None

        # Estimate pose for the first detected marker only
        # rvec/tvec shape: (1, 1, 3)
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners[:1],
            self.MARKER_LENGTH_M,
            self._camera_matrix,
            self._dist_coeffs,
        )

        rot_mat, _ = cv2.Rodrigues(rvec[0])  # Rodrigues → (3,3) rotation matrix
        q_wxyz = mat2quat(rot_mat)            # (w, x, y, z) convention

        return Pose(
            position=Point(
                x=float(tvec[0, 0, 0]),
                y=float(tvec[0, 0, 1]),
                z=float(tvec[0, 0, 2]),
            ),
            orientation=Quaternion(
                w=float(q_wxyz[0]),
                x=float(q_wxyz[1]),
                y=float(q_wxyz[2]),
                z=float(q_wxyz[3]),
            ),
        )

    # ------------------------------------------------------------------
    # Coordinate transform: camera frame → base_link
    # ------------------------------------------------------------------

    def _to_base_frame(self, pose_cam: Pose) -> Pose | None:
        """
        Transform pose_cam (in CAMERA_FRAME) to base_link via TF2.
        Also adds APPROACH_Z_OFFSET along world-Z so the gripper hovers
        above the marker rather than touching it.
        """
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", self.CAMERA_FRAME, Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            return None

        r = tf.transform.rotation
        R = quat2mat([r.w, r.x, r.y, r.z])   # camera→base rotation matrix
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])

        # Transform position from camera frame to base frame
        p_cam = np.array([
            pose_cam.position.x,
            pose_cam.position.y,
            pose_cam.position.z,
        ])
        p_base = R @ p_cam + t
        p_base[2] += self.APPROACH_Z_OFFSET  # hover above the marker

        # Compose orientations: q_base = q_cam2base * q_marker2cam
        q_marker = np.array([
            pose_cam.orientation.w,
            pose_cam.orientation.x,
            pose_cam.orientation.y,
            pose_cam.orientation.z,
        ])
        q_cam2base = np.array([r.w, r.x, r.y, r.z])
        q_base = qmult(q_cam2base, q_marker)

        return Pose(
            position=Point(
                x=float(p_base[0]),
                y=float(p_base[1]),
                z=float(p_base[2]),
            ),
            orientation=Quaternion(
                w=float(q_base[0]),
                x=float(q_base[1]),
                y=float(q_base[2]),
                z=float(q_base[3]),
            ),
        )

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
        self.get_logger().info("ArucoGripper.insert_cable() start")

        # Wait up to 5 s for camera intrinsics before doing anything
        for _ in range(50):
            if self._camera_matrix is not None:
                break
            self.sleep_for(0.1)

        if self._camera_matrix is None:
            self.get_logger().error(
                "Camera intrinsics never arrived — is /center_camera/camera_info "
                "being published? Aborting."
            )
            return False

        sleep_s = 1.0 / self.CONTROL_RATE_HZ
        deadline = time.time() + self.TASK_DURATION_S

        while time.time() < deadline:
            obs = get_observation()
            if obs is None:
                self.sleep_for(sleep_s)
                continue

            # Detect ArUco marker in the center camera image
            pose_cam = self._detect_marker_pose_in_camera(obs.center_image)
            if pose_cam is None:
                self.get_logger().info("No ArUco marker visible, searching...")
                send_feedback("searching for ArUco marker...")
                self.sleep_for(sleep_s)
                continue

            # Transform detected pose into the robot base frame
            pose_base = self._to_base_frame(pose_cam)
            if pose_base is None:
                self.sleep_for(sleep_s)
                continue

            self.get_logger().info(
                f"Marker detected — commanding gripper to "
                f"({pose_base.position.x:.3f}, "
                f"{pose_base.position.y:.3f}, "
                f"{pose_base.position.z:.3f})"
            )
            send_feedback("ArUco marker found, moving gripper...")

            # Send pose command to the gripper (uses MODE_POSITION, same as CheatCode)
            self.set_pose_target(move_robot=move_robot, pose=pose_base)
            self.sleep_for(sleep_s)

        self.get_logger().info("ArucoGripper.insert_cable() done.")
        return True
