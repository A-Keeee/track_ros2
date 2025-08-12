#include "track/track_node.hpp"


namespace track
{
    Track::Track(const rclcpp::NodeOptions & options) : Node("track_node", options)
    {
        RCLCPP_INFO(get_logger(), "Hello, TRACKOR!");

        // 初始化EKF
        ekf_ = std::make_unique<EnhancedEKF3D>(
            0.5,   // process_noise_std
            10.0,   // measurement_noise_std
            0.1,   // initial_velocity_std
            0.1,   // initial_acceleration_std
            0.1    // initial_angular_velocity_std
        );
        
        // 初始化丢失计数器和阈值
        lost_frame_count_ = 0;
        max_lost_frames_ = 30;  // 15fps，2秒 = 30帧

        raw_pose_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "/tracking/raw_pose", 10, std::bind(&Track::raw_pose_callback, this, std::placeholders::_1));

        track_pose_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/tracking/track_pose", 10);
    }

    Track::~Track()
    {

    }
    

    void Track::raw_pose_callback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
    {
        // 获取当前时间戳
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        
        // 提取3D位置
        Eigen::Vector3d measurement(msg->point.x, msg->point.y, msg->point.z);
        
        RCLCPP_INFO(get_logger(), "Received raw pose: [%.2f, %.2f, %.2f]", 
                    msg->point.x, msg->point.y, msg->point.z);

        // 检查是否为目标丢失（坐标为000）
        bool target_lost = (std::abs(msg->point.x) < 1e-6 && 
                           std::abs(msg->point.y) < 1e-6 && 
                           std::abs(msg->point.z) < 1e-6);

        try {
            if (target_lost) {
                // 目标丢失处理
                lost_frame_count_++;
                RCLCPP_WARN(get_logger(), "Target lost, frame count: %d/%d", lost_frame_count_, max_lost_frames_);
                
                if (lost_frame_count_ > max_lost_frames_) {
                    // 丢失时间过长，重置滤波器
                    RCLCPP_WARN(get_logger(), "Target lost too long, resetting EKF");
                    ekf_->reset();
                    lost_frame_count_ = 0;
                    return; // 不发布任何数据
                }
                
                if (ekf_->is_initialized()) {
                    // 使用EKF的丢失目标处理
                    auto lost_result = ekf_->handle_lost_target(timestamp);
                    if (lost_result.has_value()) {
                        auto [pred_x, pred_y, pred_z] = ekf_->predict_future_position(1.0);
                        RCLCPP_INFO(get_logger(), "Target lost - predicted 1s ahead: [%.2f, %.2f, %.2f]", 
                                    pred_x, pred_y, pred_z);
                        
                        // 发布预测位置
                        auto track_msg = geometry_msgs::msg::PointStamped();
                        track_msg.header = msg->header;
                        track_msg.point.x = pred_x;
                        track_msg.point.y = pred_y;
                        track_msg.point.z = pred_z;
                        track_pose_pub_->publish(track_msg);
                    }
                }
                return;
            }
            
            // 目标重新出现，重置丢失计数
            lost_frame_count_ = 0;
            
            if (!ekf_->is_initialized()) {
                // 首次接收数据，初始化EKF
                ekf_->initialize(measurement, timestamp);
                RCLCPP_INFO(get_logger(), "EKF initialized with first measurement");
                
                // 初始化时发布原始位置
                auto track_msg = geometry_msgs::msg::PointStamped();
                track_msg.header = msg->header;
                track_msg.point = msg->point;
                track_pose_pub_->publish(track_msg);
            } else {
                // 正常更新EKF
                ekf_->predict(timestamp);
                ekf_->update(measurement);
        
                auto [filtered_x, filtered_y, filtered_z] = ekf_->get_current_position();
                auto [vel_x, vel_y, vel_z] = ekf_->get_current_velocity();
                auto [pred_x, pred_y, pred_z] = ekf_->predict_future_position(1.0);
                
                RCLCPP_INFO(get_logger(), "Current: [%.2f, %.2f, %.2f], Vel: [%.2f, %.2f, %.2f], Pred 1s: [%.2f, %.2f, %.2f]", 
                            filtered_x, filtered_y, filtered_z, vel_x, vel_y, vel_z, pred_x, pred_y, pred_z);
                
                auto track_msg = geometry_msgs::msg::PointStamped();
                track_msg.header = msg->header;
                track_msg.point.x = pred_x;
                track_msg.point.y = pred_y;
                track_msg.point.z = pred_z;
                track_pose_pub_->publish(track_msg);
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "EKF processing error: %s", e.what());
            
            // 发生错误时，重置EKF并发布原始数据
            ekf_->reset();
            lost_frame_count_ = 0;
            auto track_msg = geometry_msgs::msg::PointStamped();
            track_msg.header = msg->header;
            track_msg.point = msg->point;
            track_pose_pub_->publish(track_msg);
        }
    }


}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(track::Track)