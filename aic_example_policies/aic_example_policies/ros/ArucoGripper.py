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

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3, Wrench
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d.quaternions import mat2quat, qmult, quat2mat
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)


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

    def to_base_frame(self, pose_cam: Pose) -> Pose | None:
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
    # Code for detection and insertion
    # ------------------------------------------------------------------ 

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )
    
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
        Build a MotionUpdate and send it to the gripper.

        This is the full implementation of Policy.set_pose_target, inlined here
        so all controller parameters are visible and tunable in one place.

        Parameters
        ----------
        pose        : target Pose for the gripper TCP (tool control point)
        frame_id    : coordinate frame that pose is expressed in
        stiffness   : diagonal of the 6-DOF Cartesian stiffness matrix
                      [x, y, z, rx, ry, rz] in N/m and N·m/rad
        damping     : diagonal of the 6-DOF Cartesian damping matrix
        wrench_feedback_gains : how much force/torque feedback is mixed in
                      (first 3 = forces, last 3 = torques)
        """
        stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]   # N/m, N·m/rad
        damping   = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]   # N·s/m, N·m·s/rad

        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            # Stiffness and damping are 6×6 diagonal matrices, stored flat (36 values)
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            # No feedforward force/torque at the tip
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            # Mix in 50 % of measured force feedback on x/y/z, none on rotation
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            # Track a position target (not a velocity target)
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    # ------------------------------------------------------------------
    # Gripper and Arm motion to specific coordinate
    # ------------------------------------------------------------------

    def gripper_move(self, move_robot, object_pose):

        #port_transform = port_tf_stamped.transform

        z_offset = 0.2

        # Over five seconds, smoothly interpolate from the current position to
        # a position above the port.
        for t in range(0, 100):
            interp_fraction = t / 100.0
            try:
                self._send_pose_to_gripper(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        object_pose,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        # Descend until the cable is inserted into the port.
        while True:
            if z_offset < -0.015:
                break

            z_offset -= 0.0005
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                self._send_pose_to_gripper(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(object_pose, z_offset=z_offset),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)

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
            pose_base = self.to_base_frame(pose_cam)
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

            # Send pose command to the gripper
            self._send_pose_to_gripper(move_robot=move_robot, pose=pose_base)
            self.sleep_for(sleep_s)

        self.get_logger().info("ArucoGripper.insert_cable() done.")
        return True
