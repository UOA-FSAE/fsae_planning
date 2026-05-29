# FSAE PLANNING STACK (SIM)
planning stack being developed testing against the fsd simulator

## Build fsae_planning
source /opt/ros/jazzy/setup.bash && colcon build --packages-select fsae_planning 2>&1

## Command to run to start the sim with bridge and autonomous stack
- Terminal1:
cd ~/repo/fsds-v2.2.0-linux
./FSDS.sh

- Terminal2:
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py UDP_control:=false

- Terminal3:
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch fsae_planning launch_planning.py


# Git Convention
## Commit Format
<type>(<scope>)/ <subject>
[blank line]
<body> [blank line] <footer> ```
Types
Type	Use
feat	new feature
fix	bug fix
docs	documentation
style	code style
refactor	code change (no fix/feat)
perf	performance
test	tests
chore	build/tooling
ci	CI/CD
Rules

    Subject: ≤50 chars, all lowercase (unless quoting)
    Body: wrap at 72, explain what/why

Examples

feat/add version control
feat(auth)/add password reset
fix(api)/handle null user


Branches

feat/xyz | fix/xyz | chore/xyz | docs/xyz | release/v1.2.3
(use kebab-case)