#!/bin/bash
# Which planner to run: centerline_planner | skidpad_planner
PLANNER="${1:-centerline_planner}"

# Launch Terminal 1: FSDS simulator
gnome-terminal -- bash -c "cd ~/fsds-v2.2.0-linux && ./FSDS.sh; exec bash" &

# Wait 5 seconds before launching the next terminals
sleep 5

# Launch Terminal 2: fsds_ros2_bridge
gnome-terminal -- bash -c "cd ~/ros2_fsd && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py UDP_control:=false; exec bash" &

# Launch Terminal 3: autonomous stack (perception bridge + planner + control + FSDS bridge)
gnome-terminal -- bash -c "cd ~/ros2_fsd && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch fsae_bringup sim.launch.py planner:=${PLANNER}; exec bash" &

# Launch Terminal 4: rosbridge websocket server (feeds the external visualiser).
# Source the sim workspace so rosbridge can (de)serialise the custom message types
# (fsae_interfaces/*, fs_msgs/*) it forwards over the websocket.
gnome-terminal -- bash -c "source /opt/ros/jazzy/setup.bash && source ~/ros2_fsd/install/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml; exec bash" &
