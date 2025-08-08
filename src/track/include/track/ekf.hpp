#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <tuple>
#include <iostream>

namespace track {
class EnhancedEKF3D {
public:
    /**
     * 增强版三维扩展卡尔曼滤波器 - 包含角速度的运动模型
     * 状态向量: [x, y, z, vx, vy, vz, ax, ay, az, theta, omega]
     */
    
    EnhancedEKF3D(
        double process_noise_std = 0.1,
        double measurement_noise_std = 0.2,
        double initial_velocity_std = 0.5,
        double initial_acceleration_std = 0.2,
        double initial_angular_velocity_std = 0.3
    );
    
    ~EnhancedEKF3D() = default;
    
    void initialize(const Eigen::Vector3d& measurement, double timestamp);
    Eigen::VectorXd predict(double timestamp);
    Eigen::VectorXd update(const Eigen::Vector3d& measurement);
    std::tuple<double, double, double> predict_future_position(std::optional<double> time_ahead = std::nullopt);
    std::tuple<double, double, double> get_current_position();
    std::tuple<double, double, double> get_current_velocity();
    std::tuple<double, double, double> get_current_acceleration();
    std::optional<std::tuple<double, double, double>> handle_lost_target(double timestamp);
    void reset();
    bool is_initialized() const { return initialized_; }
    double get_position_uncertainty();
    void set_process_noise(double std);
    void set_measurement_noise(double std);

private:
    static constexpr int STATE_DIM = 11;  // [x, y, z, vx, vy, vz, ax, ay, az, theta, omega]
    static constexpr int OBS_DIM = 3;     // [x, y, z]
    
    Eigen::VectorXd x_;  // 状态向量
    Eigen::MatrixXd P_;  // 状态协方差矩阵
    Eigen::MatrixXd Q_;  // 过程噪声协方差矩阵
    Eigen::MatrixXd R_;  // 测量噪声协方差矩阵
    Eigen::MatrixXd H_;  // 观测矩阵
    
    double last_time_;
    double last_update_time_;
    bool initialized_;
    
    double prediction_horizon_;
    int lost_count_;
    int max_lost_count_;
    
    // 辅助函数
    double normalize_angle(double angle);
};

}  // namespace track