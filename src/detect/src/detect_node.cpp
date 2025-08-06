#include "detect/detect_node.hpp"


namespace detect
{
    Detect::Detect(const rclcpp::NodeOptions & options) : Node("detect_node", options),
        detector_("/home/ake/track_ros2/src/detect/models/yolo11n-pose.torchscript", {640, 640}, 0.5f, 0.7f, 0.5f),
        fps_(0.0), frame_count_(0)
    {
        // 在控制台输出节点启动信息
        RCLCPP_INFO(get_logger(), "Hello, DETECTOR!");

        // 初始化时间戳
        last_frame_time_ = std::chrono::steady_clock::now();
        fps_start_time_ = std::chrono::steady_clock::now();

        // 创建图像发布者
        result_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/tracking/result_image", 10);

        rgb_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/color/image", 10, std::bind(&Detect::rgb_raw_callback, this, std::placeholders::_1));

        depth_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stereo/depth", 10, std::bind(&Detect::depth_raw_callback, this, std::placeholders::_1));

        // 创建发布者，用于发布3D预测位置（/hero/prediction）
        detector_pose_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/tracking/raw_pose", 10);
    }


    void Detect::rgb_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 计算帧率
        auto current_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_time_);
        last_frame_time_ = current_time;
        
        frame_count_++;
        auto fps_duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - fps_start_time_);
        if (fps_duration.count() >= 1) {
            fps_ = static_cast<double>(frame_count_) / fps_duration.count();
            frame_count_ = 0;
            fps_start_time_ = current_time;
        }

        // 处理RGB图像
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        raw_image = cv_ptr->image;

        torch::Tensor detections = detector_.detectAndVisualize(raw_image);

        persons_results = detector_.keypoints;

        // 在图像上显示帧率
        std::string fps_text = "FPS: " + std::to_string(fps_).substr(0, 5);
        cv::putText(raw_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // 发布检测结果
        auto result_msg = cv_bridge::CvImage(msg->header, "rgb8", raw_image).toImageMsg();
        result_image_pub_->publish(*result_msg);
    }

    void Detect::depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr /*msg*/)
    {
        // 处理深度图像
        // TODO: 实现深度图像处理逻辑
    }

}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(detect::Detect)