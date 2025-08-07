

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
#include "detect/reid.hpp"


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
    ~Detect();

    // 发布者
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr detector_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr result_image_pub_;


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_raw_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    void rgb_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    
    // 特征收集控制函数
    void reset_feature_collection();
    void handle_keyboard_input();

    YoloPoseDetector detector_;
    ReID reid_;

    cv::Mat raw_image;
    cv::Mat depth_image;
    std::vector<YoloPoseDetector::PersonDetection> persons_results;
    
    // 跟踪目标特征
    torch::Tensor target_feature;
    bool has_target_feature = false;
    cv::Point2f target_position; // 目标位置
    
    // 特征收集相关变量
    std::vector<torch::Tensor> collected_features; // 收集的特征向量
    std::chrono::steady_clock::time_point feature_collection_start_time; // 特征收集开始时间
    bool collecting_features = false; // 是否正在收集特征
    static constexpr int MAX_FEATURES = 5; // 最大收集特征数
    static constexpr int COLLECTION_TIME_SECONDS = 5; // 收集时间(秒)

    // 深度图像和相机内参
    bool depth_image_available = false;
    
    // 相机内参 (需要根据实际相机参数设置)
    double fx = 640.0; // 焦距x
    double fy = 640.0; // 焦距y  
    double cx = 320.0; // 主点x
    double cy = 240.0; // 主点y

    // 帧率计算相关变量
    std::chrono::steady_clock::time_point last_frame_time_;
    double fps_;
    int frame_count_;
    std::chrono::steady_clock::time_point fps_start_time_;
    
private:
    // 将2D像素坐标转换为3D世界坐标
    geometry_msgs::msg::PointStamped pixel_to_3d(const cv::Point2f& pixel, float depth);
};


} // namespace detect

#endif 