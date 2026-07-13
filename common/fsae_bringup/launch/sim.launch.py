import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


# Top-level simulator bring-up: composes perception + planning + control.
# Pick the planner with `planner:=…`; the mode wires the rest automatically.
#
#   ros2 launch fsae_bringup sim.launch.py                              # centerline_planner (default)
#   ros2 launch fsae_bringup sim.launch.py planner:=raceline_planner
#   ros2 launch fsae_bringup sim.launch.py planner:=skidpad_planner
def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('fsae_bringup'), 'launch')
    planner = LaunchConfiguration('planner')

    # Skidpad needs the whole cone map up front to reconstruct the figure-8.
    full_track = PythonExpression(
        ["'true' if '", planner, "' == 'skidpad_planner' else 'false'"]
    )

    def include(name, args):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, name)),
            launch_arguments=args.items(),
        )

    return LaunchDescription([
        DeclareLaunchArgument(
            'planner',
            default_value='centerline_planner',
            description='centerline_planner | raceline_planner | skidpad_planner'),
        include('perception.launch.py', {'full_track': full_track}),
        include('planning.launch.py',   {'planner': planner}),
        include('control.launch.py',    {'planner': planner}),
    ])
