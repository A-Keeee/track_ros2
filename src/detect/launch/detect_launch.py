import os
import sys
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription

    hero_node = ComposableNode(
        package='rm_hero',
        plugin='qianli_rm_hero::HeroNode',
        name='hero',
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='hero_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                hero_node,
            ],
            output='both',
    )
    print(container)

    return LaunchDescription([container])