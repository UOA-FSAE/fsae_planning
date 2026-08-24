import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import EqualsSubstitution, IfElseSubstitution, LaunchConfiguration
from launch_ros.actions import Node


# Control subsystem: the Stanley path-tracking controller + the FSDS command
# bridge. fsds_bridge converts cmd_vel (speed/steering) to ControlCommand and
# owns GO-gating + cone e-braking.
# The skidpad planner drives the car itself (it publishes cmd_vel directly),
# so the controller is skipped in skidpad mode; fsds_bridge still runs then
# (skidpad also uses the shared cmd_vel interface).
def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('fsae_bringup'), 'config', 'fsae_params.yaml'
    )
    planner = LaunchConfiguration('planner')
    log_csv = LaunchConfiguration('log_csv')
    log_dir = LaunchConfiguration('log_dir')
    map_path = LaunchConfiguration('map_path')
    use_precomputed_speed = LaunchConfiguration('use_precomputed_speed')
    path_map_path = LaunchConfiguration('path_map_path')
    use_precomputed_path = LaunchConfiguration('use_precomputed_path')
    v_max = LaunchConfiguration('v_max')
    v_min = LaunchConfiguration('v_min')
    stanley_gain = LaunchConfiguration('stanley_gain')
    # Effective map_path handed to the node: '' whenever the feature is
    # switched off, regardless of what map_path itself is set to -- so
    # use_precomputed_speed:=false is a reliable one-flag disable without
    # having to also clear map_path (map_path alone doubles as "where's the
    # file" and, implicitly, "is this on"; this makes "is this on" explicit).
    # IfElseSubstitution (not PythonExpression) deliberately: map_path is an
    # arbitrary filesystem path that could contain backslashes (Windows) or
    # quotes, which would corrupt/break a PythonExpression string built by
    # concatenating it into Python source text -- IfElseSubstitution passes
    # it through as data instead of evaluating it.
    effective_map_path = IfElseSubstitution(
        EqualsSubstitution(use_precomputed_speed, 'true'),
        map_path,
        '',
    )
    # Same pattern as effective_map_path, for the path-import toggle.
    effective_path_map_path = IfElseSubstitution(
        EqualsSubstitution(use_precomputed_path, 'true'),
        path_map_path,
        '',
    )
    # Skip the controller when the skidpad planner is driving the car itself.
    run_controller = UnlessCondition(EqualsSubstitution(planner, 'skidpad_planner'))

    return LaunchDescription([
        DeclareLaunchArgument('planner', default_value='centerline_planner'),
        DeclareLaunchArgument(
            'log_csv', default_value='false',
            description='Write controller CSV telemetry (e_y/e_psi/steer/...) to log_dir'),
        DeclareLaunchArgument(
            'log_dir', default_value='',
            description="Controller CSV telemetry output dir ('' -> ~/fsae_logs)"),
        DeclareLaunchArgument(
            # Default points into THIS repo's own tracks/<name>/ -- committed
            # data, not runtime-generated output, so a fresh clone of FSDS +
            # fsae_planning can drive comp_test_map_3 immediately with zero
            # setup.
            #
            # To drive a DIFFERENT (already-committed) track, don't edit this
            # line: set TRACK= in ros2/launch_all.sh, which expands to
            # map_path/path_map_path for both args at once. This default is
            # only the fallback for a bare
            # `ros2 launch fsae_bringup control.launch.py`.
            #
            # Hardcoded absolute path, not derived from __file__ or
            # get_package_share_directory(): both resolve to the INSTALLED
            # copy under ros2/install/... at runtime (confirmed: this launch
            # file is itself copied there by colcon build), which has no
            # relationship to this file's location in src/ -- there is no
            # ROS-visible path back to a source-tree sibling directory. This
            # matches WHERE launch_all.sh runs `ros2 launch` FROM (inside
            # WSL/the Docker container, not Windows) -- update this line if
            # the repo root ever moves.
            'map_path',
            default_value='/home/Formula-Student-Driverless-Simulator/ros2/src/fsae_planning/tracks/comp_test_map_3/speed_profile.csv',
            description=(
                "Path to a CSV exported from a recorded cone map, committed "
                "under this repo's own tracks/<name>/ so a fresh FSDS + "
                "fsae_planning clone can use it immediately. Has no effect "
                "unless use_precomputed_speed:=true. If the file doesn't "
                "exist, the node logs an error at startup and falls back to "
                "live curvature_speed() -- it does not crash."
            )),
        DeclareLaunchArgument(
            'use_precomputed_speed', default_value='false',
            description=(
                "true -> look up the target speed from map_path's "
                "precomputed oracle profile instead of live curvature_speed() "
                "every tick. Only valid for a track that's already been fully "
                "mapped. Set to false (default) to use live curvature_speed() "
                "behaviour regardless of map_path."
            )),
        DeclareLaunchArgument(
            # Same x,y,psi,v_target file FORMAT as map_path, but a different
            # default FILE: the track's raceline.csv rather than its
            # speed_profile.csv (the centreline).
            'path_map_path',
            default_value='/home/Formula-Student-Driverless-Simulator/ros2/src/fsae_planning/tracks/comp_test_map_3/raceline.csv',
            description=(
                "Path to a CSV exported from a recorded cone map (same "
                "this-repo tracks/<name>/ location as map_path), used as "
                "the tracked PATH (not just speed). Has no effect unless "
                "use_precomputed_path:=true."
            )),
        DeclareLaunchArgument(
            'use_precomputed_path', default_value='false',
            description=(
                "true -> track path_map_path's precomputed path instead of "
                "subscribing to the live planner's "
                "/fsae/planning/selected_trajectory -- removes "
                "centerline_planner.py from the control loop entirely, to "
                "isolate controller/plant tracking error from planner-induced "
                "path error. Only valid for a track that's already been fully "
                "mapped. On by default, matching "
                "use_precomputed_speed -- set false for the planner-in-loop "
                "diagnostic/experiment mode instead."
            )),
        # v_max/v_min/stanley_gain: overrides the controller.ros__parameters
        # block in fsae_params.yaml. Defaults match that file's current
        # values exactly, so leaving these unset on the command line changes
        # nothing; pass e.g. v_max:=3.0 for a one-off slow lap (cone-map
        # recording, characterisation runs) without editing the shared
        # config file.
        DeclareLaunchArgument(
            'v_max', default_value='15.0',
            description='m/s -- top speed on straights (overrides fsae_params.yaml controller.v_max)'),
        DeclareLaunchArgument(
            'v_min', default_value='1.5',
            description='m/s -- minimum speed through tight corners (overrides fsae_params.yaml controller.v_min)'),
        DeclareLaunchArgument(
            'stanley_gain', default_value='1.0',
            description='cross-track gain k_cte (overrides fsae_params.yaml controller.stanley_gain)'),
        Node(
            package='fsae_control',
            executable='controller',
            name='controller',
            output='screen',
            parameters=[config, {
                'log_csv': log_csv, 'log_dir': log_dir,
                'map_path': effective_map_path,
                'path_map_path': effective_path_map_path,
                'v_max': v_max, 'v_min': v_min, 'stanley_gain': stanley_gain,
            }],
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
