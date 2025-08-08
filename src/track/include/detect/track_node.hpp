#ifndef TRACK_NODE_HPP
#define TRACK_NODE_HPP

#include <iostream>
#include <memory> // 新增
#include <algorithm>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"


#include "track/ekf.hpp"


#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>



namespace track
{

class Track : public rclcpp::Node
{
public:
    Track(const rclcpp::NodeOptions & options);
    ~Track();

    // 发布者
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr track_pose_pub_;

    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr detector_pose_sub_;


    void raw_pose_callback(const sensor_msgs::msg::Image::SharedPtr msg);


};


} // namespace detect

#endif 