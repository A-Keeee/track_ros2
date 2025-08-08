import os
import sys
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    track_node = ComposableNode(
        package='track',
        plugin='track::Track',
        name='track_node',
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='track_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                track_node,
            ],
            output='both',
    )
    print(container)

    return LaunchDescription([container])