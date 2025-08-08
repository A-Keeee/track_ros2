// reid.cpp
#include "detect/reid.hpp"
#include <iostream>

ReID::ReID(const std::string& model_path, bool use_gpu)
    : device_(torch::kCPU) {
    // 决定使用CPU还是GPU
    if (use_gpu && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        std::cout << "ReID模型正在使用 CUDA 设备进行推理。" << std::endl;
    } else {
        std::cout << "ReID模型正在使用 CPU 设备进行推理。" << std::endl;
    }

    try {
        // 加载序列化的PyTorch模型
        module_ = torch::jit::load(model_path);
        // 将模型移动到指定的设备
        module_.to(device_);
        // 将模型设置为评估模式
        module_.eval();
    } catch (const c10::Error& e) {
        std::cerr << "加载ReID模型失败: " << e.what() << std::endl;
        throw; // 抛出异常，让调用者知道初始化失败
    }
}

ReID::~ReID() {}

torch::Tensor ReID::preprocess(const cv::Mat& image) const {
    // 确保输入图像是 BGR 格式
    if (image.empty()) {
        throw std::runtime_error("输入的图像为空!");
    }

    cv::Mat resized_image;
    // 1. 调整图像尺寸至模型输入尺寸 (宽 x 高)
    cv::resize(image, resized_image, cv::Size(input_width_, input_height_));

    // 2. 将 BGR 转换为 RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // 3. 将图像数据类型转换为浮点型并归一化到 [0, 1]
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // 4. 将 cv::Mat 转换为 torch::Tensor
    // from_blob 不会复制数据，效率很高
    auto tensor_image = torch::from_blob(float_image.data, {1, input_height_, input_width_, 3});

    // 5. 调整维度顺序从 HWC 到 CHW (PyTorch模型需要 NCHW 格式)
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    // 6. 标准化 (Normalization)
    torch::data::transforms::Normalize<> normalize(mean_, std_);
    tensor_image = normalize(tensor_image);

    return tensor_image;
}

torch::Tensor ReID::preprocess_optimized(const cv::Mat& image) const {
    // 确保输入图像是 BGR 格式
    if (image.empty()) {
        throw std::runtime_error("输入的图像为空!");
    }

    cv::Mat resized_image;
    // 1. 调整图像尺寸至模型输入尺寸 (宽 x 高)
    cv::resize(image, resized_image, cv::Size(input_width_, input_height_));

    // 2. 将 BGR 转换为 RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // 3. 将图像数据类型转换为浮点型并归一化到 [0, 1]
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // 4. 直接创建tensor并拷贝数据到目标设备
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto tensor_image = torch::from_blob(float_image.data, {1, input_height_, input_width_, 3}, torch::kFloat32)
                        .to(options.device()).clone(); // clone确保数据拷贝

    // 5. 调整维度顺序从 HWC 到 CHW (PyTorch模型需要 NCHW 格式)
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    // 6. 标准化 (Normalization) - 在目标设备上进行
    auto mean_tensor = torch::tensor({0.485, 0.456, 0.406}, options).view({1, 3, 1, 1});
    auto std_tensor = torch::tensor({0.229, 0.224, 0.225}, options).view({1, 3, 1, 1});
    tensor_image = (tensor_image - mean_tensor) / std_tensor;

    return tensor_image;
}

torch::Tensor ReID::extract_feature(const cv::Mat& image) {
    // 使用 NoGradGuard 来禁用梯度计算，可以加速推理并减少内存消耗
    torch::NoGradGuard no_grad;

    // 1. 预处理图像 - 直接在目标设备上处理
    torch::Tensor input_tensor = preprocess_optimized(image);

    // 2. 执行前向推理
    // 我们将输入Tensor包装在一个vector中，因为module.forward()需要一个IValue的向量
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    auto output = module_.forward(inputs).toTensor();

    // 3. (可选但推荐) 对输出特征进行L2归一化
    // 这使得特征向量的比较可以使用余弦相似度，且更加稳定
    output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // 只在需要时移回CPU
    if (device_.is_cuda()) {
        return output.to(torch::kCPU);
    }
    return output;
}

torch::Tensor ReID::extract_features_batch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        return torch::empty({0, 2048}); // 假设特征维度是2048
    }

    torch::NoGradGuard no_grad;
    
    // 预处理所有图像并堆叠成批
    std::vector<torch::Tensor> preprocessed_images;
    preprocessed_images.reserve(images.size());
    
    for (const auto& image : images) {
        if (!image.empty()) {
            auto processed = preprocess_optimized(image);
            preprocessed_images.push_back(processed.squeeze(0)); // 移除batch维度
        }
    }
    
    if (preprocessed_images.empty()) {
        return torch::empty({0, 2048});
    }
    
    // 将所有图像堆叠成一个批次
    auto batch_tensor = torch::stack(preprocessed_images, 0);
    
    // 批量推理
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);
    
    auto output = module_.forward(inputs).toTensor();
    
    // L2归一化
    output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    
    // 只在需要时移回CPU
    if (device_.is_cuda()) {
        return output.to(torch::kCPU);
    }
    return output;
}