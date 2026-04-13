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


import numpy as np
import cv2

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

from datetime import datetime
from pathlib import Path


class WaveArm(Policy):

    DATA_ROOT: str = "/home/administrato/dev/ws_aic/data"

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("WaveArm.__init__()")

    @staticmethod
    def save_image(image_msg, path: Path) -> None:
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
        self.get_logger().info(f"WaveArm.insert_cable() enter. Task: {task}")

        # --------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_parts = [
            p for p in [task.cable_name, task.plug_name, task.port_name, timestamp]
            if p
        ]
        scene_name = "__".join(scene_parts) if scene_parts else timestamp
        scene_dir = Path(self.DATA_ROOT) / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"WaveAndCollect: saving scene to '{scene_dir}'")
        idx = 0
        # --------------------------------

        start_time = self.time_now()
        timeout = Duration(seconds=10.0)
        send_feedback("waving the arm around")
        while (self.time_now() - start_time) < timeout:
            self.sleep_for(0.25)
            observation = get_observation()

            if observation is None:
                self.get_logger().info("No observation received.")
                continue

            t = (
                observation.center_image.header.stamp.sec
                + observation.center_image.header.stamp.nanosec / 1e9
            )
            self.get_logger().info(f"observation time - kuku: {t}")

            # ---------------------------------------
            # Save left / center / right images
            prefix      = f"point_{idx:03d}"
            left_name   = f"{scene_dir}\\{prefix}_left.png"
            center_name = f"{scene_dir}\\{prefix}_center.png"
            right_name  = f"{scene_dir}\\{prefix}_right.png"

            self.save_image(observation.left_image,   left_name)
            self.save_image(observation.center_image, center_name)
            self.save_image(observation.right_image,  right_name)
            self.get_logger().info(f"Saved image: {center_name}")

            idx += 1
            # ---------------------------------------            

            loop_seconds = 5.0
            loop_fraction = (t % loop_seconds) / loop_seconds
            y_scale = 2 * loop_fraction
            if y_scale > 1.0:
                y_scale = 2.0 - y_scale
            y_scale -= 1.0  # y_scale will move linearly between [-1..1] and back.

            # Move the arm along a line, while looking down at the task board.
            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(
                    position=Point(x=-0.4, y=0.45 + 0.3 * y_scale, z=0.25),
                    orientation=Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
                ),
            )



        self.get_logger().info("WaveArm.insert_cable() exiting...")
        return True
