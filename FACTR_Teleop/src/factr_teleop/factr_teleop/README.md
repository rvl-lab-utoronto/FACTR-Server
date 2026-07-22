As of Jun 18 2026, the code for gravity compensation, friction compensation etc. within the factr_teleop's FACTRTeleop class has been commented out, it should be uncommented when we manage to send joint torque messages from the Rizon arms back to Factr.

As of Jul 13 2026:
    factr_teleop_dual_base.py and factr_rizon_dual_board.py are intended to be used together to control one arm with two U2D2 boards.
    The relay provides one typed WebSocket per arm. Each stream carries readings
    and diagnostics; gravity-comp commands/status remain HTTP routes.
    Suggested procedures:
        Initialize terminals:
            `cd ~/FACTR-Server/FACTR_Teleop`
            `source /opt/ros/humble/setup.bash`
            `source install/setup.bash`
        Start arms control:
            `ros2 run factr_teleop factr_joint_pub` for the left arm
            `ros2 run factr_teleop frdb` for the right arm
        To start the integrated relay:
            `/usr/bin/python3 -m src.factr_fastapi.factr_fastapi.factr_api`
        WebSockets:
            `ws://localhost:5000/ws/left`
            `ws://localhost:5001/ws/right`

    Next step:
        1. put the arm into a ready position
        2. start grav comp
        3. start teleoperation (for Flexiv client: if the websocket channels return None, that means the arm is not yet ready)

It's possible that factr_joint_publisher.py was never called.

