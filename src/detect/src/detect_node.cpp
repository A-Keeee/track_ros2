#include "detect/detect_node.hpp"


namespace detect
{
    Detect::Detect(const rclcpp::NodeOptions & options) : Node("detect_node", options),
        detector_("/home/ake/track_ros2/src/detect/models/yolo11n-pose.torchscript", {640, 640}, 0.5f, 0.7f, 0.5f),
        reid_("/home/ake/track_ros2/src/detect/models/ReID_resnet50_ibn_a.torchscript", true),
        fps_(0.0), frame_count_(0), collecting_features(false), has_target_feature(false)
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

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/color/camera_info", 10, std::bind(&Detect::camera_info_callback, this, std::placeholders::_1));

        // 创建发布者，用于发布3D预测位置（/hero/prediction）
        detector_pose_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/tracking/raw_pose", 10);
        
        // 创建OpenCV窗口用于显示图像和接收键盘输入
        cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);
    }

    Detect::~Detect()
    {
        // 清理OpenCV窗口
        cv::destroyAllWindows();
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

        int best_match_index = -1; // 将变量定义移到外部


        if (has_target_feature && !persons_results.empty()) {
            float min_distance = std::numeric_limits<float>::max();

            for (size_t i = 0; i < persons_results.size(); ++i) {
                const auto& person = persons_results[i];
                if (person.x < 0 || person.y < 0 || person.x + person.w > raw_image.cols || person.y + person.h > raw_image.rows) {
                    continue; // 跳过无效的边界框
                }
                cv::Mat person_img = raw_image(cv::Rect(person.x, person.y, person.w, person.h)).clone();
                torch::Tensor feature = reid_.extract_feature(person_img);
                float distance = torch::nn::functional::pairwise_distance(feature, target_feature).item<float>();
                // std::cout << "Distance to target feature: " << distance << std::endl;
                if (distance < min_distance && distance < 1.0f) { 
                    min_distance = distance;
                    best_match_index = i;
                }
            }

            if (best_match_index != -1) {
                // 在图像上绘制最佳匹配的边界框
                const auto& best_person = persons_results[best_match_index];
                cv::rectangle(raw_image, cv::Point(best_person.x, best_person.y), cv::Point(best_person.x + best_person.w, best_person.y + best_person.h), cv::Scalar(255, 0, 0), 5);
                cv::putText(raw_image, "Target", cv::Point(best_person.x, best_person.y - 10), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 5);
                target_position = cv::Point2f((best_person.kpt_5.x + best_person.kpt_6.x + best_person.kpt_11.x + best_person.kpt_12.x) / 4.0, (best_person.kpt_5.y + best_person.kpt_6.y + best_person.kpt_11.y + best_person.kpt_12.y) / 4.0);
            }
        } else if (!has_target_feature && !persons_results.empty()) {
            // 如果还没有目标特征，开始收集特征
            const auto& first_person = persons_results[0];
            if (first_person.x >= 0 && first_person.y >= 0 && 
                first_person.x + first_person.w <= raw_image.cols && 
                first_person.y + first_person.h <= raw_image.rows) {
                
                auto current_time = std::chrono::steady_clock::now();
                
                // 如果还没开始收集特征，初始化收集过程
                if (!collecting_features) {
                    collecting_features = true;
                    feature_collection_start_time = current_time;
                    collected_features.clear();
                    RCLCPP_INFO(get_logger(), "Started collecting target features...");
                }
                
                // 检查是否在收集时间内
                auto collection_duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - feature_collection_start_time);
                
                if (collection_duration.count() <= COLLECTION_TIME_SECONDS && collected_features.size() < MAX_FEATURES) {
                    // 在收集时间内且还没收集够特征数量，继续收集
                    cv::Mat person_img = raw_image(cv::Rect(first_person.x, first_person.y, first_person.w, first_person.h)).clone();
                    torch::Tensor feature = reid_.extract_feature(person_img);
                    collected_features.push_back(feature.clone());
                    
                    RCLCPP_INFO(get_logger(), "Collected feature %d/%d", (int)collected_features.size(), MAX_FEATURES);
                    
                    // 在图像上显示收集状态
                    cv::rectangle(raw_image, cv::Point(first_person.x, first_person.y), 
                                cv::Point(first_person.x + first_person.w, first_person.y + first_person.h), 
                                cv::Scalar(0, 255, 255), 3); // 黄色框表示正在收集
                    cv::putText(raw_image, "Collecting " + std::to_string(collected_features.size()) + "/" + std::to_string(MAX_FEATURES), 
                              cv::Point(first_person.x, first_person.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
                } 
                
                // 如果收集完成或超时，计算平均特征
                if (collected_features.size() >= MAX_FEATURES || collection_duration.count() > COLLECTION_TIME_SECONDS) {
                    if (collected_features.size() > 0) {
                        // 计算平均特征
                        torch::Tensor sum_feature = collected_features[0].clone();
                        for (size_t i = 1; i < collected_features.size(); ++i) {
                            sum_feature += collected_features[i];
                        }
                        target_feature = sum_feature / static_cast<float>(collected_features.size());
                        has_target_feature = true;
                        collecting_features = false;
                        
                        RCLCPP_INFO(get_logger(), "Feature collection completed! Used %d features for averaging.", (int)collected_features.size());
                        
                        // 在图像上显示完成状态
                        cv::rectangle(raw_image, cv::Point(first_person.x, first_person.y), 
                                    cv::Point(first_person.x + first_person.w, first_person.y + first_person.h), 
                                    cv::Scalar(0, 255, 0), 3); // 绿色框表示完成
                        cv::putText(raw_image, "Target Locked!", cv::Point(first_person.x, first_person.y - 10), 
                                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }   
        
        // 如果找到目标并且有深度信息，发布3D坐标
        if (best_match_index != -1 && depth_image_available) {
            // 获取目标位置的深度值
            int pixel_x = static_cast<int>(target_position.x);
            int pixel_y = static_cast<int>(target_position.y);
            
            // 确保像素坐标在深度图范围内
            if (pixel_x >= 0 && pixel_x < depth_image.cols && 
                pixel_y >= 0 && pixel_y < depth_image.rows) {
                
                // 获取深度值（假设深度图是16位，单位为毫米）
                uint16_t depth_raw = depth_image.at<uint16_t>(pixel_y, pixel_x);
                float depth_meters = depth_raw / 1000.0f; // 转换为米
                
                // 只有当深度值有效时才发布3D坐标
                if (depth_meters > 0.1f && depth_meters < 10.0f) { // 合理的深度范围
                    geometry_msgs::msg::PointStamped point_3d = pixel_to_3d(target_position, depth_meters);
                    point_3d.header.stamp = msg->header.stamp;
                    point_3d.header.frame_id = "camera_frame"; 
                    cv::Point3f xyz_point = cv::Point3f(point_3d.point.x, point_3d.point.y, point_3d.point.z);
                    point_3d.point.x = xyz_point.z;
                    point_3d.point.y = -xyz_point.x;
                    point_3d.point.z = -xyz_point.y; 
                    detector_pose_pub_->publish(point_3d);
                    
                    // 在图像上显示3D坐标信息
                    const auto& best_person = persons_results[best_match_index];
                    std::string coord_text = "3D: (" + 
                        std::to_string(point_3d.point.x).substr(0, 4) + ", " +
                        std::to_string(point_3d.point.y).substr(0, 4) + ", " +
                        std::to_string(point_3d.point.z).substr(0, 4) + ")";
                    cv::putText(raw_image, coord_text, cv::Point(best_person.x, best_person.y - 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 255, 0), 5);
                }
            }
        }

        // 在图像上显示帧率
        std::string fps_text = "FPS: " + std::to_string(fps_).substr(0, 5);
        cv::putText(raw_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // 在图像上显示操作提示
        cv::putText(raw_image, "Press 'R' to reset target", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::Mat test_image = raw_image.clone(); 
        cv::cvtColor(test_image, test_image, cv::COLOR_RGB2BGR);
        // 显示图像
        cv::imshow("Detection Result", test_image);

        // 处理键盘输入
        handle_keyboard_input();

        // 发布检测结果
        auto result_msg = cv_bridge::CvImage(msg->header, "rgb8", raw_image).toImageMsg();
        result_image_pub_->publish(*result_msg);
    }

    void Detect::depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 处理深度图像
        try {
            cv_bridge::CvImagePtr cv_ptr;
            
            // 根据深度图像编码类型进行转换
            if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
                // 如果是32位浮点深度图，需要转换为16位
                cv_ptr->image.convertTo(depth_image, CV_16UC1, 1000.0); // 假设输入是米，转换为毫米
                depth_image_available = true;
                return;
            } else {
                RCLCPP_WARN(get_logger(), "Unsupported depth image encoding: %s", msg->encoding.c_str());
                return;
            }
            
            depth_image = cv_ptr->image.clone();
            depth_image_available = true;
            
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            depth_image_available = false;
        }
    }



    geometry_msgs::msg::PointStamped Detect::pixel_to_3d(const cv::Point2f& pixel, float depth)
    {
        geometry_msgs::msg::PointStamped point_3d;
        
        // 使用针孔相机模型将2D像素坐标转换为3D世界坐标
        // X = (u - cx) * Z / fx
        // Y = (v - cy) * Z / fy
        // Z = depth
        
        point_3d.point.x = (pixel.x - cx) * depth / fx;
        point_3d.point.y = (pixel.y - cy) * depth / fy;
        point_3d.point.z = depth;
        
        return point_3d;
    }

    void Detect::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // 从相机信息消息中获取内参
        fx = msg->k[0]; // K[0,0]
        fy = msg->k[4]; // K[1,1]
        cx = msg->k[2]; // K[0,2]
        cy = msg->k[5]; // K[1,2]
        
        RCLCPP_INFO_ONCE(get_logger(), "Camera intrinsics received: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", 
                         fx, fy, cx, cy);
    }

    void Detect::reset_feature_collection()
    {
        collecting_features = false;
        has_target_feature = false;
        collected_features.clear();
        RCLCPP_INFO(get_logger(), "Feature collection reset! Ready to collect new target.");
    }

    void Detect::handle_keyboard_input()
    {
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'r' || key == 'R') {
            reset_feature_collection();
        }
    }

}

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(detect::Detect)