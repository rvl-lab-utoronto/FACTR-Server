# Read-only FACTR leader joint publisher.
#
# Reads the leader arm's DYNAMIXEL servos and publishes CALIBRATED joint positions on
# /joint_pos_{side} — WITHOUT enabling torque, so the arm stays fully backdrivable
# (move it by hand during data collection). It publishes raw hardware state; managed
# teleop receives its arm calibration and directions from DFC instead.
#
# Launch once per leader arm (arm_index selects side + config, like factr_api):
#   ros2 run factr_teleop factr_joint_pub --ros-args -p arm_index:=0   # left
#   ros2 run factr_teleop factr_joint_pub --ros-args -p arm_index:=1   # right
#
# The ONLY bus write this node ever issues is set_torque_mode(False) at startup; every
# timer tick is a pure read (Fast Sync Read). It never calls set_torque_mode(True), so
# no motor is ever energized. Calibration (_get_dynamixel_offsets) and the read+publish
# math (get_leader_joint_states) are copied verbatim from FACTRTeleop so the numbers are
# identical to full teleop.

import os
import subprocess

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from python_utils.utils import get_workspace_root
from factr_teleop.dynamixel.driver import DynamixelDriver
from factr_teleop.factr_teleop import find_ttyusb

"""This is never executed"""


class FactrJointPublisher(Node):
    """Read one leader with torque off and publish untouched Dynamixel state."""

    def __init__(self):
        super().__init__("factr_joint_publisher")
        arm_index = self.declare_parameter("arm_index", 0).get_parameter_value().integer_value
        self.side = "left" if arm_index == 0 else "right"

        cfg_name = f"factr_rizon_{self.side}.yaml"
        cfg_path = os.path.join(
            get_workspace_root(), f"src/factr_teleop/factr_teleop/configs/{cfg_name}"
        )
        with open(cfg_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dt = 1.0 / self.config["controller"]["frequency"]
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        # ---- driver init: torque stays OFF, arm backdrivable ----
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]

        # FTDI latency timer must be 1 ms to sustain the read rate (default 16 ms ~ 60 Hz).
        ttyusbx = find_ttyusb(self.dynamixel_port)
        result = subprocess.run(
            f"cat /sys/bus/usb-serial/devices/{ttyusbx}/latency_timer",
            shell=True, capture_output=True, text=True, check=True,
        )
        if int(result.stdout) != 1:
            raise RuntimeError(
                f"Set the latency timer of {ttyusbx} to 1:\n"
                f"  echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyusbx}/latency_timer"
            )

        first_id = int(self.config["dynamixel"].get("first_id", 1))
        joint_ids = np.arange(self.num_motors) + first_id
        self.driver = DynamixelDriver(
            joint_ids,
            self.servo_types,
            self.dynamixel_port,
        )
        # Torque stays off (read-only). The driver ctor already attempted this; retry
        # but TOLERATE failure (a servo may report an error flag or already be off) so
        # we still proceed to READ positions rather than aborting the node.
        try:
            self.driver.set_torque_mode(False)
        except Exception as e:
            self.get_logger().warning(f"set_torque_mode(False) failed ({e}); reading anyway")

        self.pub = self.create_publisher(JointState, f"/joint_pos_{self.side}", 10)
        self.timer = self.create_timer(self.dt, self.publish_cb)
        self.get_logger().info(
            f"FACTR joint publisher [{self.side}]: reading {self.num_motors} servos "
            f"(torque OFF, backdrivable) -> /joint_pos_{self.side} @ {1.0 / self.dt:.0f} Hz"
        )

    def publish_cb(self):
        try:
            raw_pos, raw_vel = self.driver.get_positions_and_velocities()
        except RuntimeError:
            # Transient Dynamixel bus timeout (e.g. -3001) at 500 Hz: skip this tick and
            # keep the node alive rather than aborting; the next read usually succeeds.
            self._read_fails = getattr(self, "_read_fails", 0) + 1
            if self._read_fails % 200 == 1:
                self.get_logger().warning(f"transient servo read failure (count={self._read_fails}); skipping tick")
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i}" for i in range(self.num_arm_joints)] + ["gripper"]
        msg.position = np.asarray(raw_pos, dtype=float).tolist()
        msg.velocity = np.asarray(raw_vel, dtype=float).tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FactrJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Read-only handle: nothing to de-energize, but be explicit and tidy.
        try:
            node.driver.set_torque_mode(False)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
