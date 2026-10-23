import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Simulator perception stand-in: FSDS oracle map + odom → /fsae/* inputs.
# `full_track:=true` publishes the entire map every frame (used by skidpad).
#
# fov_mode:=camera emulates the kart's chain instead: sim_perception is the
# camera only (cone_detection frames), and the kart's own cone_mapper_shim
# turns those frames into /fsae/slam/*_track. The shim reads
# /cone_detection_fused on the kart (the LiDAR-refined frames); here it is
# remapped to the camera frames, since the sim has no LiDAR fusion.
#
# cone_memory (a longer-lived cone map in its own process, see
# fsae_planning/cone_memory.py) runs when cone_memory:=true or when the
# planner reads it (planner_cone_source:=memory). It reads the same frames,
# so its view of "should have seen it" follows camera_hfov_deg/camera_range.
def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('fsae_bringup'), 'config', 'fsae_params.yaml'
    )
    full_track = LaunchConfiguration('full_track')
    fov_mode = LaunchConfiguration('fov_mode')
    camera_hfov_deg = LaunchConfiguration('camera_hfov_deg')
    camera_range = LaunchConfiguration('camera_range')
    camera_dropout = LaunchConfiguration('camera_dropout')
    camera_fixed_miss = LaunchConfiguration('camera_fixed_miss')
    camera_both_colours_gate = LaunchConfiguration('camera_both_colours_gate')
    cone_memory = LaunchConfiguration('cone_memory')
    planner_cone_source = LaunchConfiguration('planner_cone_source')

    camera_on = PythonExpression(
        ["'", fov_mode, "' == 'camera' and '", full_track, "'.lower() != 'true'"])
    memory_on = PythonExpression(
        ["'", cone_memory, "'.lower() == 'true' or '", planner_cone_source, "' == 'memory'"])
    detections = [('/cone_detection_fused', '/fsae/perception/cone_detection')]

    return LaunchDescription([
        DeclareLaunchArgument('full_track', default_value='false',
                              description='publish the whole cone map instead of a FOV window'),
        # Camera-mode overrides (see fsae_params.yaml sim_perception). Defaults
        # match that file, so leaving them unset changes nothing.
        DeclareLaunchArgument('fov_mode', default_value='window',
                              description="'window' (look_radius/box crop) | 'camera' "
                                          '(kart camera + cone_mapper_shim)'),
        DeclareLaunchArgument('camera_hfov_deg', default_value='110.0',
                              description='camera mode: horizontal field of view (deg)'),
        DeclareLaunchArgument('camera_range', default_value='10.0',
                              description='camera mode: detection range (m) from the car'),
        DeclareLaunchArgument('camera_dropout', default_value='0.0',
                              description='camera mode: per-cone, per-frame miss probability'),
        DeclareLaunchArgument('camera_fixed_miss', default_value='0.0',
                              description='camera mode: fraction of blue/yellow cones never detected'),
        DeclareLaunchArgument('camera_both_colours_gate', default_value='false',
                              description="camera mode: drop frames without both colours, "
                                          "like the kart's cone_detection"),
        DeclareLaunchArgument('cone_memory', default_value='false',
                              description='run cone_memory (publishes /fsae/cone_memory/*)'),
        DeclareLaunchArgument('planner_cone_source', default_value='slam',
                              description="'slam' | 'memory' -- only decides whether cone_memory "
                                          'must run here (see planning.launch.py)'),
        Node(
            package='fsae_sim_perception',
            executable='sim_perception',
            name='sim_perception',
            output='screen',
            parameters=[config, {
                'full_track': full_track,
                'fov_mode': fov_mode,
                'camera_dropout': camera_dropout,
                'camera_fixed_miss': camera_fixed_miss,
                'camera_hfov_deg': camera_hfov_deg,
                'camera_range': camera_range,
                'camera_both_colours_gate': camera_both_colours_gate,
            }],
        ),
        Node(
            package='fsae_planning',
            executable='cone_mapper_shim',
            name='cone_mapper_shim',
            output='screen',
            parameters=[config],
            remappings=detections,
            condition=IfCondition(camera_on),
        ),
        Node(
            package='fsae_planning',
            executable='cone_memory',
            name='cone_memory',
            output='screen',
            parameters=[config, {
                'view_hfov_deg': camera_hfov_deg,
                'view_range': camera_range,
            }],
            remappings=detections,
            condition=IfCondition(memory_on),
        ),
    ])
