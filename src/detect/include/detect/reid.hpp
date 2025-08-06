// reid.hpp
#pragma once

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <torch/data/transforms.h> 
#include <torch/nn.h>              
#include <torch/csrc/api/include/torch/cuda.h> 

class ReID {
public:
    /**
     * @brief 构造函数，加载ReID模型并初始化
     * @param model_path TorchScript模型文件的路径
     * @param use_gpu 是否使用GPU进行推理
     */
    ReID(const std::string& model_path, bool use_gpu = true);

    /**
     * @brief 析构函数
     */
    ~ReID();

    /**
     * @brief 对单张行人图像提取ReID特征
     * @param image 输入的OpenCV图像 (cv::Mat)，图像中应只包含一个裁剪好的行人
     * @return 返回一个torch::Tensor，即提取到的特征向量
     */
    torch::Tensor extract_feature(const cv::Mat& image);

private:
    /**
     * @brief 对图像进行预处理，以满足模型输入要求
     * @param image 输入的OpenCV图像
     * @return 返回一个预处理后的Tensor
     */
    torch::Tensor preprocess(const cv::Mat& image) const;

    torch::jit::script::Module module_; // 加载的TorchScript模型
    torch::Device device_;             // 推理设备 (CPU or CUDA)

    // ReID模型通常的输入尺寸
    const int input_height_ = 256;
    const int input_width_ = 128;

    // ImageNet的标准化参数，大多数ReID模型都使用这个
    const std::vector<double> mean_ = {0.485, 0.456, 0.406};
    const std::vector<double> std_ = {0.229, 0.224, 0.225};
};