import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Control subsystem: a path-tracking controller (cmd_vel) + the FSDS command bridge.
# The controller is selectable with `controller:=stanley|mpc` (both publish the
# same cmd_vel interface; fsds_bridge converts it identically).  The skidpad
# planner drives the car itself (it publishes cmd_vel directly), so the
# controller is skipped in skidpad mode; fsds_bridge always runs.
def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('fsae_bringup'), 'config', 'fsae_params.yaml'
    )
    planner = LaunchConfiguration('planner')
    controller = LaunchConfiguration('controller')
    run_controller = IfCondition(
        PythonExpression(["'", planner, "' != 'skidpad_planner'"])
    )
    # Map the friendly name to the package entry point; node name stays
    # 'controller' either way so both read the `controller:` params block.
    controller_exec = PythonExpression(
        ["'mpc_controller' if '", controller, "' == 'mpc' else 'controller'"]
    )

    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value='centerline_planner'),
        DeclareLaunchArgument(
            'controller', default_value='stanley',
            description='stanley | mpc — path-tracking controller to run'),
        Node(
            package='fsae_control',
            executable=controller_exec,
            name='controller',
            output='screen',
            parameters=[config],
            condition=run_controller,
        ),
        Node(
            package='fsae_control',
            executable='fsds_bridge',
            name='fsds_bridge',
            output='screen',
            parameters=[config],
        ),
    ])
