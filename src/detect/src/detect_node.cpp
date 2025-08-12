#include "detect/detect_node.hpp"


namespace detect
{
    Detect::Detect(const rclcpp::NodeOptions & options) : Node("detect_node", options),
        detector_("/home/ake/track_ros2/src/detect/models/yolo11n-pose.torchscript", {640, 640}, 0.5f, 0.7f, 0.5f),
        reid_("/home/ake/track_ros2/src/detect/models/ReID_resnet50_ibn_a.torchscript", true),
        fps_(0.0), frame_count_(0), collecting_features(false), has_target_feature(false)
    {
        RCLCPP_INFO(get_logger(), "Hello, DETECTOR!");

        last_frame_time_ = std::chrono::steady_clock::now();
        fps_start_time_ = std::chrono::steady_clock::now();

        result_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/tracking/result_image", 10);

        rgb_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/color/image", 10, std::bind(&Detect::rgb_raw_callback, this, std::placeholders::_1));

        depth_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/stereo/depth", 10, std::bind(&Detect::depth_raw_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/color/camera_info", 10, std::bind(&Detect::camera_info_callback, this, std::placeholders::_1));

        detector_pose_pub_ = create_publisher<geometry_msgs::msg::PointStamped>("/tracking/raw_pose", 10);
        
        cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);
    }

    Detect::~Detect()
    {
        cv::destroyAllWindows();
    }


    void Detect::rgb_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
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

        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        raw_image = cv_ptr->image;

        torch::Tensor detections = detector_.detectAndVisualize(raw_image);

        persons_results = detector_.keypoints;

        int best_match_index = -1; 


        if (has_target_feature && !persons_results.empty()) {
            float min_distance = std::numeric_limits<float>::max();
            
            // 收集所有有效的人员图像进行批处理
            std::vector<cv::Mat> person_images;
            std::vector<size_t> valid_indices;
            
            for (size_t i = 0; i < persons_results.size(); ++i) {
                const auto& person = persons_results[i];
                if (person.x < 0 || person.y < 0 || person.x + person.w > raw_image.cols || person.y + person.h > raw_image.rows) {
                    continue; 
                }
                cv::Mat person_img = raw_image(cv::Rect(person.x, person.y, person.w, person.h)).clone();
                person_images.push_back(person_img);
                valid_indices.push_back(i);
            }
            
            
            if (!person_images.empty()) {
                torch::Tensor batch_features = reid_.extract_features_batch(person_images);
                
                // 计算每个人与目标的距离
                for (size_t j = 0; j < batch_features.size(0); ++j) {
                    torch::Tensor feature = batch_features[j].unsqueeze(0);
                    float distance = torch::nn::functional::pairwise_distance(feature, target_feature).item<float>();
                    if (distance < min_distance && distance < 1.0f) { 
                        min_distance = distance;
                        best_match_index = valid_indices[j];
                    }
                }
            }

            if (best_match_index != -1) {
                const auto& best_person = persons_results[best_match_index];
                cv::rectangle(raw_image, cv::Point(best_person.x, best_person.y), cv::Point(best_person.x + best_person.w, best_person.y + best_person.h), cv::Scalar(255, 0, 0), 5);
                cv::putText(raw_image, "Target", cv::Point(best_person.x, best_person.y - 10), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 0, 0), 5);
                target_position = cv::Point2f((best_person.kpt_5.x + best_person.kpt_6.x + best_person.kpt_11.x + best_person.kpt_12.x) / 4.0, (best_person.kpt_5.y + best_person.kpt_6.y + best_person.kpt_11.y + best_person.kpt_12.y) / 4.0);
            }
        } else if (!has_target_feature && !persons_results.empty()) {
            const auto& first_person = persons_results[0];
            if (first_person.x >= 0 && first_person.y >= 0 && 
                first_person.x + first_person.w <= raw_image.cols && 
                first_person.y + first_person.h <= raw_image.rows) {
                
                auto current_time = std::chrono::steady_clock::now();
                
                if (!collecting_features) {
                    collecting_features = true;
                    feature_collection_start_time = current_time;
                    collected_features.clear();
                    RCLCPP_INFO(get_logger(), "Started collecting target features...");
                }
                
                auto collection_duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - feature_collection_start_time);

                if (collection_duration.count() <= COLLECTION_TIME_SECONDS && collected_features.size() < MAX_FEATURES && collection_duration.count() - collected_features.size()*0.6 > 0) {
                    cv::Mat person_img = raw_image(cv::Rect(first_person.x, first_person.y, first_person.w, first_person.h)).clone();
                    torch::Tensor feature = reid_.extract_feature(person_img);
                    collected_features.push_back(feature.clone());
                    
                    RCLCPP_INFO(get_logger(), "Collected feature %d/%d", (int)collected_features.size(), MAX_FEATURES);
                    
                    cv::rectangle(raw_image, cv::Point(first_person.x, first_person.y), 
                                cv::Point(first_person.x + first_person.w, first_person.y + first_person.h), 
                                cv::Scalar(0, 255, 255), 3); 
                    cv::putText(raw_image, "Collecting " + std::to_string(collected_features.size()) + "/" + std::to_string(MAX_FEATURES), 
                              cv::Point(first_person.x, first_person.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
                } 
                
                if (collected_features.size() >= MAX_FEATURES || collection_duration.count() > COLLECTION_TIME_SECONDS) {
                    if (collected_features.size() > 0) {
                        torch::Tensor sum_feature = collected_features[0].clone();
                        for (size_t i = 1; i < collected_features.size(); ++i) {
                            sum_feature += collected_features[i];
                        }
                        target_feature = sum_feature / static_cast<float>(collected_features.size());
                        has_target_feature = true;
                        collecting_features = false;
                        
                        RCLCPP_INFO(get_logger(), "Feature collection completed! Used %d features for averaging.", (int)collected_features.size());
                        
                        cv::rectangle(raw_image, cv::Point(first_person.x, first_person.y), 
                                    cv::Point(first_person.x + first_person.w, first_person.y + first_person.h), 
                                    cv::Scalar(0, 255, 0), 3);
                        cv::putText(raw_image, "Target Locked!", cv::Point(first_person.x, first_person.y - 10), 
                                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                    }
                }
            }
        }   
        
        if (best_match_index != -1 && depth_image_available) {
            int pixel_x = static_cast<int>(target_position.x);
            int pixel_y = static_cast<int>(target_position.y);
            
            if (pixel_x >= 0 && pixel_x < depth_image.cols && 
                pixel_y >= 0 && pixel_y < depth_image.rows) {
                
                uint16_t depth_raw = depth_image.at<uint16_t>(pixel_y, pixel_x);
                float depth_meters = depth_raw / 1000.0f; 
                
                if (depth_meters > 0.1f && depth_meters < 10.0f) { 
                    geometry_msgs::msg::PointStamped point_3d = pixel_to_3d(target_position, depth_meters);
                    point_3d.header.stamp = msg->header.stamp;
                    point_3d.header.frame_id = "camera_frame"; 
                    cv::Point3f xyz_point = cv::Point3f(point_3d.point.x, point_3d.point.y, point_3d.point.z);
                    point_3d.point.x = xyz_point.z;
                    point_3d.point.y = -xyz_point.x;
                    point_3d.point.z = -xyz_point.y; 
                    detector_pose_pub_->publish(point_3d);
                    
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


        std::string fps_text = "FPS: " + std::to_string(fps_).substr(0, 3);
        cv::putText(raw_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(raw_image, "Press 'R' to reset target", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        cv::Mat test_image = raw_image.clone(); 
        cv::cvtColor(test_image, test_image, cv::COLOR_BGR2RGB);
        cv::imshow("Detection Result", test_image);

        handle_keyboard_input();

        auto result_msg = cv_bridge::CvImage(msg->header, "bgr8", raw_image).toImageMsg();
        result_image_pub_->publish(*result_msg);
    }

    void Detect::depth_raw_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            cv_bridge::CvImagePtr cv_ptr;
            
            if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
                cv_ptr->image.convertTo(depth_image, CV_16UC1, 1000.0); 
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