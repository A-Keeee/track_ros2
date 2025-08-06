#include "detect/detector.hpp"

using torch::indexing::Slice;
using torch::indexing::None;

// 构造函数
YoloPoseDetector::YoloPoseDetector(const std::string& model_path, 
                                   const std::vector<int>& input_size,
                                   float conf_thres,
                                   float iou_thres,
                                   float kpt_conf_thres)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      input_size_(input_size),
      conf_thres_(conf_thres),
      iou_thres_(iou_thres),
      kpt_conf_thres_(kpt_conf_thres) {
    
    std::cout << "当前设备: " << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    
    // 模型加载
    auto load_start = std::chrono::high_resolution_clock::now();
    model_ = torch::jit::load(model_path, device_);
    model_.eval();
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "模型加载耗时: " << std::fixed << std::setprecision(3) << load_time << " ms" << std::endl;

    // GPU预热 (Warm-up)
    if (device_.is_cuda()) {
        std::cout << "正在进行GPU预热..." << std::endl;
        torch::Tensor dummy_input = torch::randn({1, 3, input_size_[0], input_size_[1]}, device_);
        for (int i = 0; i < 5; ++i) {
            model_.forward({dummy_input});
        }
        torch::cuda::synchronize();
        std::cout << "预热完成!" << std::endl;
    }
}

// 检测函数
torch::Tensor YoloPoseDetector::detect(const cv::Mat& image) {
    // 图像预处理
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    torch::Tensor input_tensor = preprocessImage(image);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    // 推理
    std::vector<torch::jit::IValue> inputs{input_tensor};
    
    if(device_.is_cuda()) torch::cuda::synchronize(device_.index());
    auto inference_start = std::chrono::high_resolution_clock::now();

    torch::Tensor output_gpu = model_.forward(inputs).toTensor();

    if(device_.is_cuda()) torch::cuda::synchronize(device_.index());
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    torch::Tensor output = output_gpu.cpu();

    // 后处理
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    auto detections = nonMaxSuppressionPose(output, conf_thres_, iou_thres_)[0];
    
    // 坐标缩放到原始图像尺寸
    scaleCoords(input_size_, detections, {image.rows, image.cols});
    auto postprocess_end = std::chrono::high_resolution_clock::now();

    // 性能计时输出
    double pre_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    double infer_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double post_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    
    // std::cout << "[YOLO-Pose(" << (device_.is_cuda() ? "CUDA" : "CPU") << ")]: "
    //           << std::fixed << std::setprecision(3)
    //           << pre_time << "ms pre-process, "
    //           << infer_time << "ms inference, "
    //           << post_time << "ms post-process" << std::endl;

    return detections;
}

// 检测并可视化
torch::Tensor YoloPoseDetector::detectAndVisualize(cv::Mat& image) {
    torch::Tensor detections = detect(image);
    visualizeResults(image, detections);
    return detections;
}

// 预处理图像
torch::Tensor YoloPoseDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat processed_image;
    letterbox(image, processed_image, input_size_);
    cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);

    torch::Tensor input_tensor = torch::from_blob(
        processed_image.data, 
        {processed_image.rows, processed_image.cols, 3}, 
        torch::kByte
    ).to(device_)
    .to(torch::kFloat32)
    .div(255)
    .permute({2, 0, 1})
    .unsqueeze(0);
    
    return input_tensor;
}

// 可视化结果
void YoloPoseDetector::visualizeResults(cv::Mat& image, const torch::Tensor& detections) {
    keypoints.clear();
    for (int i = 0; i < detections.size(0); ++i) {
        auto det = detections[i]; // Shape: [56]
        
        // 绘制边界框
        int x1 = det[0].item().toInt();
        int y1 = det[1].item().toInt();
        int x2 = det[2].item().toInt();
        int y2 = det[3].item().toInt();
        float conf = det[4].item().toFloat();

        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        std::string label = "person " + std::to_string(conf).substr(0, 4);
        cv::putText(image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 提取并绘制关键点和骨骼
        torch::Tensor kpts = det.slice(0, 5, 5 + kpt_num_ * 3).reshape({kpt_num_, 3});


        PersonDetection person;
        person.x = x1;
        person.y = y1;
        person.w = x2 - x1;
        person.h = y2 - y1;
        person.kpt_5 = cv::Point2f(kpts[5][0].item().toInt(), kpts[5][1].item().toInt());
        person.kpt_6 = cv::Point2f(kpts[6][0].item().toInt(), kpts[6][1].item().toInt());
        person.kpt_11 = cv::Point2f(kpts[11][0].item().toInt(), kpts[11][1].item().toInt());
        person.kpt_12 = cv::Point2f(kpts[12][0].item().toInt(), kpts[12][1].item().toInt());
        keypoints.push_back(person);
        
        // 绘制骨骼
        for (size_t j = 0; j < skeleton_.size(); ++j) {
            int kpt_idx1 = skeleton_[j].first;
            int kpt_idx2 = skeleton_[j].second;

            float conf1 = kpts[kpt_idx1][2].item().toFloat();
            float conf2 = kpts[kpt_idx2][2].item().toFloat();

            if (conf1 > kpt_conf_thres_ && conf2 > kpt_conf_thres_) {
                int x_coord1 = kpts[kpt_idx1][0].item().toInt();
                int y_coord1 = kpts[kpt_idx1][1].item().toInt();
                int x_coord2 = kpts[kpt_idx2][0].item().toInt();
                int y_coord2 = kpts[kpt_idx2][1].item().toInt();
                cv::line(image, cv::Point(x_coord1, y_coord1), cv::Point(x_coord2, y_coord2), limb_colors_[j % limb_colors_.size()], 2);
            }
        }

        // 绘制关键点
        for (int k = 0; k < kpt_num_; ++k) {
            float kpt_conf = kpts[k][2].item().toFloat();
            if (kpt_conf > kpt_conf_thres_) {
                int x_coord = kpts[k][0].item().toInt();
                int y_coord = kpts[k][1].item().toInt();
                cv::circle(image, cv::Point(x_coord, y_coord), 5, kpt_colors_[k], -1);
            }
        }
    }
}


// 图像预处理：计算缩放比例
float YoloPoseDetector::generateScale(const cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;
    int target_h = target_size[0];
    int target_w = target_size[1];
    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

// 图像预处理：对图像进行 letterbox 缩放和填充
float YoloPoseDetector::letterbox(const cv::Mat& input_image, cv::Mat& output_image, const std::vector<int>& target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generateScale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
}

// 坐标转换：将 (center_x, center_y, width, height) 格式转为 (x1, y1, x2, y2)
torch::Tensor YoloPoseDetector::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

// C++ 实现的 NMS
torch::Tensor YoloPoseDetector::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();
    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);
    auto order_t = std::get<1>(scores.sort(true, 0, true));
    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();
    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];
        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);
            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}


/**
 * @brief 对YOLO-Pose模型的输出进行非极大值抑制（NMS）
 * @param prediction 模型的原始输出, shape: [bs, 56, num_proposals]
 * @param conf_thres 置信度阈值
 * @param iou_thres IOU阈值
 * @param max_det 最大检测数量
 * @return 返回经过NMS后的检测结果列表，每个元素的shape为 [num_dets, 56]
 */
std::vector<torch::Tensor> YoloPoseDetector::nonMaxSuppressionPose(torch::Tensor& prediction, float conf_thres, float iou_thres, int max_det) {
    // Pose模型只有一个类别'person', 所以 nc=1, 但我们实际上不使用类别
    // 格式为 [box(4), conf(1), kpts(17*3=51)]
    
    // 1. 根据置信度进行初步过滤
    // prediction[..., 4] 是bbox的置信度
    auto xc = prediction.index({Slice(), 4}) > conf_thres;

    // 2. 将 cx,cy,w,h 格式的box转为 x1,y1,x2,y2
    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({"...", Slice(None, 4)}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < prediction.size(0); i++) {
        output.push_back(torch::zeros({0, 56}, prediction.device()));
    }

    // 3. 逐张图片进行处理
    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({xc[xi]}); // 应用置信度过滤

        if (x.size(0) == 0) {
            continue;
        }
        
        // 4. 对边界框执行NMS
        auto boxes = x.index({Slice(), Slice(0, 4)});
        auto scores = x.index({Slice(), 4});
        auto i = nms(boxes, scores, iou_thres); // NMS
        i = i.index({Slice(None, max_det)}); // 限制最大检测数量
        
        output[xi] = x.index({i});
    }

    return output;
}

/**
 * @brief 将检测到的坐标（边界框和关键点）从模型输入尺寸映射回原始图像尺寸
 * @param img1_shape 模型输入图像的尺寸 {h, w}, e.g., {640, 640}
 * @param coords 检测结果张量, shape: [num_dets, 56]
 * @param img0_shape 原始图像的尺寸 {h, w}
 * @return 返回缩放后的坐标张量
 */
torch::Tensor YoloPoseDetector::scaleCoords(const std::vector<int>& img1_shape, torch::Tensor& coords, const std::vector<int>& img0_shape) {
    // 1. 计算缩放增益和填充量
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad_x = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad_y = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    // 2. 移除填充
    // 缩放边界框
    coords.index_put_({"...", 0}, coords.index({"...", 0}) - pad_x); // x1
    coords.index_put_({"...", 2}, coords.index({"...", 2}) - pad_x); // x2
    coords.index_put_({"...", 1}, coords.index({"...", 1}) - pad_y); // y1
    coords.index_put_({"...", 3}, coords.index({"...", 3}) - pad_y); // y2
    
    // 缩放关键点
    // coords的5~55列是关键点，格式为 [x, y, conf, x, y, conf, ...]
    // x坐标在 5, 8, 11, ... 位置
    // y坐标在 6, 9, 12, ... 位置
    coords.index_put_({"...", Slice(5, None, 3)}, coords.index({"...", Slice(5, None, 3)}) - pad_x);
    coords.index_put_({"...", Slice(6, None, 3)}, coords.index({"...", Slice(6, None, 3)}) - pad_y);

    // 3. 缩放到原始尺寸
    coords.index_put_({"...", Slice(0, 4)}, coords.index({"...", Slice(0, 4)}).div(gain));
    coords.index_put_({"...", Slice(5, None, 3)}, coords.index({"...", Slice(5, None, 3)}).div(gain));
    coords.index_put_({"...", Slice(6, None, 3)}, coords.index({"...", Slice(6, None, 3)}).div(gain));

    return coords;
}



// // 示例main函数，展示如何使用YoloPoseDetector类
// int main() {
//     try {
//         // 配置路径
//         std::string model_path = "/home/ake/yolo_gpu_detect/test/yolo11n-pose.torchscript";
//         std::string image_path = "/home/ake/yolo_gpu_detect/test/image.jpg";

//         // 创建检测器实例
//         YoloPoseDetector detector(model_path, {640, 640}, 0.5f, 0.7f, 0.5f);

//         // 读取图像
//         cv::Mat image = cv::imread(image_path);
//         if (image.empty()) {
//             std::cerr << "无法读取图像: " << image_path << std::endl;
//             return -1;
//         }

//         // 执行检测并可视化
//         torch::Tensor detections = detector.detectAndVisualize(image);

//         // 显示结果
//         cv::imshow("Pose Detection Results", image);
//         cv::imwrite("pose_result.jpg", image); // 保存结果图
//         cv::waitKey(0);

//         std::cout << "检测到 " << detections.size(0) << " 个人体" << std::endl;

//     } catch (const c10::Error& e) {
//         std::cerr << "Torch错误: " << e.what() << std::endl;
//         return -1;
//     } catch (const cv::Exception& e) {
//         std::cerr << "OpenCV错误: " << e.what() << std::endl;
//         return -1;
//     } catch (const std::exception& e) {
//         std::cerr << "标准错误: " << e.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }