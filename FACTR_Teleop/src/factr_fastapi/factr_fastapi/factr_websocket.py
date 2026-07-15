# ROS2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data

import asyncio
import threading
import time

# Websockets
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
import json
 
MSG_TYPE = JointState
 
# Websockets config
WS_HOST = "127.0.0.1"
WS_PORT = 5001

# ROS2 subscriptions
TOPIC_LEFT = "/joint_pos_left" # left arm
TOPIC_RIGHT = "/joint_pos_right"
BROADCAST_HZ = 200               
QOS = qos_profile_sensor_data   
 
# Latest joint positions
value_lock = threading.Lock()
joint_pos_latest = {"left": None, "right": None}



CLIENTS = set()
 
 
def msg_to_dict(msg) -> dict:
    """Convert the ROS2 message into a JSON"""
    return {
        "position": list(msg.position),
    }
 
 
def build_combined() -> str:
    """Combine left and right arm readings"""
    with value_lock:
        left = joint_pos_latest["left"]
        right = joint_pos_latest["right"]
    payload = {
        "left": left,
        "right": right,
    }
    return json.dumps(payload)
 
 
# ROS2 subscriber
class JointBridgeNode(Node):
    def __init__(self):
        super().__init__("ros2_websocket_wrapper")
        self.create_subscription(MSG_TYPE, TOPIC_LEFT, self._make_cb("left"), QOS)
        self.create_subscription(MSG_TYPE, TOPIC_RIGHT, self._make_cb("right"), QOS)
        self.get_logger().info(f"Subscribed to {TOPIC_LEFT} and {TOPIC_RIGHT}")
 
    def _make_cb(self, side):
        def _cb(msg):
            data = msg_to_dict(msg)
            with value_lock:
                joint_pos_latest[side] = data
        return _cb
 
 
def ros_spin(node):
    """Spin the node; returns when rclpy is shut down from the main thread."""
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
 
 
# Websocket server
async def ws_handler(websocket):
    """Register a client and keep the connection open (we only push data out)."""
    CLIENTS.add(websocket)
    peer = getattr(websocket, "remote_address", None)
    print(f"[ws] client connected: {peer}  (total {len(CLIENTS)})")
    try:
        # We don't expect inbound messages; iterating drains them and lets us
        # notice when the client disconnects.
        async for _ in websocket:
            pass
    except ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"[ws] client disconnected: {peer}  (total {len(CLIENTS)})")
 
 
async def broadcast_loop():
    """Push the merged state to every connected client at BROADCAST_HZ."""
    period = 1.0 / BROADCAST_HZ
    while True:
        if CLIENTS:
            message = build_combined()
            dead = set()
            for ws in list(CLIENTS):
                try:
                    await ws.send(message)
                except Exception:
                    dead.add(ws)          # slow/broken client -> drop it
            for ws in dead:
                CLIENTS.discard(ws)
        await asyncio.sleep(period)
 
 
async def main_async():
    async with serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[ws] serving on ws://{WS_HOST}:{WS_PORT}")
        await broadcast_loop()            # runs until cancelled
 
 
def main():
    rclpy.init()
    node = JointBridgeNode()
 
    # Spin ROS in a background daemon thread; asyncio owns the main thread.
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()
 
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=2.0)
 
 
if __name__ == "__main__":
    main()

