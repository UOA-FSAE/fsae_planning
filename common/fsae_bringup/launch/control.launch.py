import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Control subsystem: the Stanley controller (cmd_vel) + the FSDS command bridge.
# The skidpad planner drives the car itself (it publishes cmd_vel directly), so
# the Stanley controller is skipped in skidpad mode; fsds_bridge always runs.
def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('fsae_bringup'), 'config', 'fsae_params.yaml'
    )
    planner = LaunchConfiguration('planner')
    run_controller = IfCondition(
        PythonExpression(["'", planner, "' != 'skidpad_planner'"])
    )

    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value='centerline_planner'),
        Node(
            package='fsae_control',
            executable='controller',
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
