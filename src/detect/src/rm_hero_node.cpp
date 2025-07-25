#include "rm_hero_node.hpp"


namespace qianli_rm_hero
{
    HeroNode::HeroNode(const rclcpp::NodeOptions & options) : Node("rm_hero_node", options),
    frame_count_(0),
    last_time_(this->now())
    
    {
        // 在控制台输出节点启动信息
        RCLCPP_INFO(get_logger(), "Hello, QianLi RM Hero!");

        // 创建订阅者，接受英雄坐标
        hero_pose_sub_ = this->create_subscription<auto_aim_interfaces::msg::ReceiveSerial>(
            "/angle/init", 10, std::bind(&HeroNode::Hero_pose_callback, this, std::placeholders::_1));


        // 创建发布者，用于发布3D预测位置（/hero/prediction）
        hero_pose_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/hero/prediction", 10);

        // 初始化tf2缓存和监听器，用于将预测的3D坐标转换到不同的坐标系
        tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
        tf2_buffer_->setCreateTimerInterface(timer_interface);
        tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);


    }



    void HeroNode::Hero_pose_callback(const auto_aim_interfaces::msg::ReceiveSerial msg)
    {   
        

        // 新增帧率计算逻辑
        auto current_time = this->now();
        frame_count_++;
        double elapsed = (current_time - last_time_).seconds();
        
        if (elapsed >= 1.0) {
            double fps = frame_count_ / elapsed;
            RCLCPP_INFO(get_logger(), "[FPS] Current: %.2f", fps);
            frame_count_ = 0;
            last_time_ = current_time;
        }


        // 创建消息并填充当前英雄坐标
        geometry_msgs::msg::PointStamped point_msg;
        point_msg.header.frame_id = "odom";
        point_msg.header.stamp = msg.header.stamp;
        point_msg.point.x = msg.hero_pose_x;
        point_msg.point.y = msg.hero_pose_y;



        // 梯高下 z = 200mm;
        // 梯高上 z = 600mm;
        // 公路 z = 200mm;
        if (detect_color == 1) {//打蓝方
            if (msg.hero_pose_x > blue_x1 && msg.hero_pose_x < blue_x2 && msg.hero_pose_y > blue_y1 && msg.hero_pose_y < blue_y2) {
                point_msg.point.z = 600.0; 
            } else {
                point_msg.point.z = 200.0; 
            }
        } else {//打红方
            if (msg.hero_pose_x > red_x1 && msg.hero_pose_x < red_x2 && msg.hero_pose_y > red_y1 && msg.hero_pose_y < red_y2) {
                point_msg.point.z = 600.0; 
            } else {
                point_msg.point.z = 200.0; 
            }
        }

        point_msg.point.z += hero_gimbal_height_; // 加上云台高度

        //此时坐标为英雄当前实时坐标
        // 进行坐标转换需要将坐标原点移动到当前英雄坐标系原点
        if (detect_color == 1){
            point_msg.point.x = blue_base_point_.x - point_msg.point.x;
            point_msg.point.y = blue_base_point_.y - point_msg.point.y;
            point_msg.point.z = base_height_ - point_msg.point.z;
        } else {
            point_msg.point.x = red_base_point_.x - point_msg.point.x;
            point_msg.point.y = red_base_point_.y - point_msg.point.y;
            point_msg.point.z = base_height_ - point_msg.point.z;
        }


        try {
            hero_pose_pub_->publish(point_msg);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(get_logger(), "HERO无法正确发布坐标%s", ex.what());
        }
    }
} // namespace qianli_rm_hero

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(qianli_rm_hero::HeroNode)