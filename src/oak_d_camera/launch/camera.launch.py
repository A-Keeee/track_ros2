#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
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

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_tf_prefix_cmd)
    ld.add_action(declare_lrcheck_cmd)
    ld.add_action(declare_extended_cmd)
    ld.add_action(declare_subpixel_cmd)
    ld.add_action(declare_confidence_cmd)
    ld.add_action(declare_LRchecktresh_cmd)

    # Add the camera node
    ld.add_action(camera_node)

    return ld
