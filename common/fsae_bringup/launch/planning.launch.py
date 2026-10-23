import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


# Planning subsystem: launches ONE planner, selected by the `planner` arg.
#   centerline_planner  - barebone cone-wall centreline (no localisation)
#   skidpad_planner     - figure-8 characterisation (special track type)
# Node name == executable == config key, so params load from fsae_params.yaml.
def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('fsae_bringup'), 'config', 'fsae_params.yaml'
    )
    planner = LaunchConfiguration('planner')
    look_radius = LaunchConfiguration('look_radius')
    plan_horizon = LaunchConfiguration('plan_horizon')
    debug_viz = LaunchConfiguration('debug_viz')
    use_orange = LaunchConfiguration('use_orange')
    one_sided = LaunchConfiguration('one_sided')
    planner_cone_source = LaunchConfiguration('planner_cone_source')

    run_centerline = IfCondition(
        PythonExpression(["'", planner, "' == 'centerline_planner' and '",
                          planner_cone_source, "' != 'memory'"])
    )
    # Same planner, cones from cone_memory: its four track inputs remapped to
    # /fsae/cone_memory/* (no code change -- the node only knows /fsae/slam/*).
    run_centerline_memory = IfCondition(
        PythonExpression(["'", planner, "' == 'centerline_planner' and '",
                          planner_cone_source, "' == 'memory'"])
    )
    memory_remaps = [(f'/fsae/slam/{t}', f'/fsae/cone_memory/{t}')
                     for t in ('left_track', 'right_track', 'big_orange_track', 'small_orange_track')]
    run_skidpad = IfCondition(
        PythonExpression(["'", planner, "' == 'skidpad_planner'"])
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'planner',
            default_value='centerline_planner',
            description='centerline_planner | skidpad_planner'),
        # look_radius/plan_horizon: overrides fsae_params.yaml's
        # centerline_planner.look_radius/plan_horizon (centerline_planner
        # only -- skidpad_planner never declares these, so they're passed
        # only to the centerline_planner Node below). Defaults match that
        # file's current values, so leaving these unset changes nothing.
        # Raise both together (they're kept equal by convention --
        # see boundary._WALL_PLAN_HORIZON's comment) to test whether a
        # corner's speed-profile issue is the planner's own crop window
        # rather than sensing range -- pair with full_track:=true on
        # sim.launch.py to rule out perception range entirely first.
        DeclareLaunchArgument(
            'look_radius', default_value='25.0',
            description='m -- cone visibility radius for planning '
                        '(overrides fsae_params.yaml centerline_planner.look_radius)'),
        DeclareLaunchArgument(
            'plan_horizon', default_value='25.0',
            description='m -- arc-length horizon the published centreline is clamped to '
                        '(overrides fsae_params.yaml centerline_planner.plan_horizon)'),
        DeclareLaunchArgument(
            'debug_viz', default_value='false',
            description='centerline_planner only -- publish /fsae/planning/debug/* '
                        '(midpoints, wall mesh, Delaunay triangulation). Read once at '
                        'startup, so it must be set here rather than via ros2 param set.'),
        DeclareLaunchArgument(
            'use_orange', default_value='true',
            description='centerline_planner only -- plan with the big orange start-line '
                        'cones (overrides fsae_params.yaml centerline_planner.use_orange; '
                        'false is the rollback). Read once at startup.'),
        DeclareLaunchArgument(
            'planner_cone_source', default_value='slam',
            description="centerline_planner only -- 'slam' reads /fsae/slam/*_track as on "
                        "the kart; 'memory' reads cone_memory's /fsae/cone_memory/*_track."),
        DeclareLaunchArgument(
            'one_sided', default_value='true',
            description='centerline_planner only -- extend the path along the one '
                        'visible boundary when the gate walk ends (overrides '
                        'fsae_params.yaml centerline_planner.one_sided; false is the '
                        'rollback).'),
        Node(
            package='fsae_planning',
            executable=planner,
            name=planner,
            output='screen',
            parameters=[config, {
                'look_radius': look_radius,
                'plan_horizon': plan_horizon,
                'debug_viz': debug_viz,
                'use_orange': use_orange,
                'one_sided': one_sided,
            }],
            condition=run_centerline,
        ),
        Node(
            package='fsae_planning',
            executable=planner,
            name=planner,
            output='screen',
            parameters=[config, {
                'look_radius': look_radius,
                'plan_horizon': plan_horizon,
                'debug_viz': debug_viz,
                'use_orange': use_orange,
                'one_sided': one_sided,
            }],
            remappings=memory_remaps,
            condition=run_centerline_memory,
        ),
        Node(
            package='fsae_planning',
            executable=planner,
            name=planner,
            output='screen',
            parameters=[config],
            condition=run_skidpad,
        ),
    ])
