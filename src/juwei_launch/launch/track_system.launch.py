#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
import launch_ros.actions


def generate_launch_description():
    
    # Configuration for camera parameters
    tf_prefix    = LaunchConfiguration('tf_prefix',     default='oak')
    lrcheck      = LaunchConfiguration('lrcheck',       default=True)
    extended     = LaunchConfiguration('extended',      default=False)
    subpixel     = LaunchConfiguration('subpixel',      default=False)
    confidence   = LaunchConfiguration('confidence',    default=200)
    LRchecktresh = LaunchConfiguration('LRchecktresh',  default=5)

    # Declare launch arguments
    declare_tf_prefix_cmd = DeclareLaunchArgument(
        'tf_prefix',
        default_value=tf_prefix,
        description='The prefix for the tf frames.')

    declare_lrcheck_cmd = DeclareLaunchArgument(
        'lrcheck',
        default_value=lrcheck,
        description='Left-right check for stereo depth.')

    declare_extended_cmd = DeclareLaunchArgument(
        'extended',
        default_value=extended,
        description='Use extended disparity.')

    declare_subpixel_cmd = DeclareLaunchArgument(
        'subpixel',
        default_value=subpixel,
        description='Use subpixel mode.')

    declare_confidence_cmd = DeclareLaunchArgument(
        'confidence',
        default_value=confidence,
        description='Confidence threshold for depth.')

    declare_LRchecktresh_cmd = DeclareLaunchArgument(
        'LRchecktresh',
        default_value=LRchecktresh,
        description='Left-right check threshold.')

    # Camera node (RGB + Depth only)
    camera_node = launch_ros.actions.Node(
        package='oak_d_camera',
        executable='camera_node',
        name='camera',
        parameters=[{
            'tf_prefix': tf_prefix,
            'lrcheck': lrcheck,
            'extended': extended,
            'subpixel': subpixel,
            'confidence': confidence,
            'LRchecktresh': LRchecktresh,
        }],
        output='screen'
    )

    # Detection node as composable component
    detect_node = ComposableNode(
        package='detect',
        plugin='detect::Detect',
        name='detect_node',
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    track_node = ComposableNode(
        package='track',
        plugin='track::Track',
        name='track_node',
        extra_arguments=[{'use_intra_process_comms': True}],
    )



    # Composable node container for detection
    detect_container = ComposableNodeContainer(
        name='detect_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            detect_node,
            track_node
        ],
        output='both',
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_tf_prefix_cmd)
    ld.add_action(declare_lrcheck_cmd)
    ld.add_action(declare_extended_cmd)
    ld.add_action(declare_subpixel_cmd)
    ld.add_action(declare_confidence_cmd)
    ld.add_action(declare_LRchecktresh_cmd)

    # Add the camera node (start first)
    ld.add_action(camera_node)
    
    # Add a small delay before starting detection to ensure camera is ready
    ld.add_action(
        TimerAction(
            period=2.0,  # 2 second delay
            actions=[detect_container]
        )
    )

    return ld
