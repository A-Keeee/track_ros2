

#ifndef RM_HERO_NODE_HPP
#define RM_HERO_NODE_HPP

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


#include "detect/detector.hpp"


#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>



namespace detect
{

class Detect : public rclcpp::Node
{
public:
    Detect(const rclcpp::NodeOptions & options);

    // 发布者
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr detector_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_image_pub_;


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_raw_sub_;

    void rgb_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg);

    YoloPoseDetector detector_;

    cv::Mat raw_image;
    std::vector<YoloPoseDetector::PersonDetection> persons_results;


    // 帧率计算相关变量
    std::chrono::steady_clock::time_point last_frame_time_;
    double fps_;
    int frame_count_;
    std::chrono::steady_clock::time_point fps_start_time_;
};


} // namespace detect

#endif 