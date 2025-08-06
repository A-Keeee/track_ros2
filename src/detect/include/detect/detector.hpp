#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <iostream>

/**
 * @brief YOLO-Pose检测器类，封装了人体姿态检测的完整流程
 */
class YoloPoseDetector {
public:
    /**
     * @brief 构造函数
     * @param model_path 模型文件路径
     * @param input_size 输入图像尺寸，默认为{640, 640}
     * @param conf_thres 置信度阈值，默认为0.5
     * @param iou_thres IOU阈值，默认为0.7
     * @param kpt_conf_thres 关键点置信度阈值，默认为0.5
     */
    YoloPoseDetector(const std::string& model_path, 
                     const std::vector<int>& input_size = {640, 640},
                     float conf_thres = 0.5f,
                     float iou_thres = 0.7f,
                     float kpt_conf_thres = 0.5f);

    /**
     * @brief 析构函数
     */
    ~YoloPoseDetector() = default;

    /**
     * @brief 检测图像中的人体姿态
     * @param image 输入图像
     * @return 检测结果的张量
     */
    torch::Tensor detect(const cv::Mat& image);

    /**
     * @brief 检测并可视化结果
     * @param image 输入图像（会被修改以绘制检测结果）
     * @return 检测结果的张量
     */
    torch::Tensor detectAndVisualize(cv::Mat& image);

    /**
     * @brief 设置置信度阈值
     */
    void setConfThreshold(float conf_thres) { conf_thres_ = conf_thres; }

    /**
     * @brief 设置IOU阈值
     */
    void setIouThreshold(float iou_thres) { iou_thres_ = iou_thres; }

    /**
     * @brief 设置关键点置信度阈值
     */
    void setKptConfThreshold(float kpt_conf_thres) { kpt_conf_thres_ = kpt_conf_thres; }
    
    struct PersonDetection {
        // 边界框
        float x;
        float y;
        float w;
        float h;

        // 关键点（第5、6、11、12号，分别是左肩、右肩、左髋、右髋）
        cv::Point2f kpt_5;  // 左肩
        cv::Point2f kpt_6;  // 右肩
        cv::Point2f kpt_11; // 左髋
        cv::Point2f kpt_12; // 右髋
    };

    std::vector<PersonDetection> keypoints; // 存储检测到的人的关键点信息



private:
    // 私有成员变量
    torch::jit::script::Module model_;
    torch::Device device_;
    std::vector<int> input_size_;
    float conf_thres_;
    float iou_thres_;
    float kpt_conf_thres_;
    static const int kpt_num_ = 17; // COCO数据集的关键点数量

    // COCO 17个关键点的骨骼连接关系
    const std::vector<std::pair<int, int>> skeleton_ = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
        {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}
    };

    // 骨骼连接线的颜色
    const std::vector<cv::Scalar> limb_colors_ = {
        {255, 0, 0}, {255, 75, 0}, {255, 150, 0}, {255, 225, 0}, {0, 255, 0},
        {0, 255, 75}, {0, 255, 150}, {0, 255, 225}, {0, 0, 255}, {75, 0, 255},
        {150, 0, 255}, {225, 0, 255}, {255, 0, 255}, {255, 0, 150}, {255, 0, 75}
    };

    // 关键点的颜色
    const std::vector<cv::Scalar> kpt_colors_ = {
        {255, 0, 0}, {255, 75, 0}, {255, 150, 0}, {255, 225, 0}, {0, 255, 0},
        {0, 255, 75}, {0, 255, 150}, {0, 255, 225}, {0, 0, 255}, {75, 0, 255},
        {150, 0, 255}, {225, 0, 255}, {255, 0, 255}, {255, 0, 150}, {255, 0, 75},
        {0, 75, 255}, {0, 150, 255}
    };

    // 私有方法
    /**
     * @brief 图像预处理：计算缩放比例
     */
    float generateScale(const cv::Mat& image, const std::vector<int>& target_size);

    /**
     * @brief 图像预处理：对图像进行 letterbox 缩放和填充
     */
    float letterbox(const cv::Mat& input_image, cv::Mat& output_image, const std::vector<int>& target_size);

    /**
     * @brief 坐标转换：将 (center_x, center_y, width, height) 格式转为 (x1, y1, x2, y2)
     */
    torch::Tensor xywh2xyxy(const torch::Tensor& x);

    /**
     * @brief C++ 实现的 NMS
     */
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);

    /**
     * @brief 对YOLO-Pose模型的输出进行非极大值抑制（NMS）
     * @param prediction 模型的原始输出, shape: [bs, 56, num_proposals]
     * @param conf_thres 置信度阈值
     * @param iou_thres IOU阈值
     * @param max_det 最大检测数量
     * @return 返回经过NMS后的检测结果列表，每个元素的shape为 [num_dets, 56]
     */
    std::vector<torch::Tensor> nonMaxSuppressionPose(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);

    /**
     * @brief 将检测到的坐标（边界框和关键点）从模型输入尺寸映射回原始图像尺寸
     * @param img1_shape 模型输入图像的尺寸 {h, w}, e.g., {640, 640}
     * @param coords 检测结果张量, shape: [num_dets, 56]
     * @param img0_shape 原始图像的尺寸 {h, w}
     * @return 返回缩放后的坐标张量
     */
    torch::Tensor scaleCoords(const std::vector<int>& img1_shape, torch::Tensor& coords, const std::vector<int>& img0_shape);

    /**
     * @brief 在图像上绘制检测结果
     * @param image 要绘制的图像
     * @param detections 检测结果
     */
    void visualizeResults(cv::Mat& image, const torch::Tensor& detections);

    /**
     * @brief 预处理输入图像
     * @param image 输入图像
     * @return 预处理后的张量
     */
    torch::Tensor preprocessImage(const cv::Mat& image);
};
