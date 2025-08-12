#include "track/ekf.hpp"
#include <cmath>
#include <algorithm>

namespace track {
// ================================================================================================
// EnhancedEKF3D Implementation
// ================================================================================================

EnhancedEKF3D::EnhancedEKF3D(
    double process_noise_std,
    double measurement_noise_std,
    double initial_velocity_std,
    double initial_acceleration_std,
    double initial_angular_velocity_std)
    : x_(STATE_DIM), P_(STATE_DIM, STATE_DIM), Q_(STATE_DIM, STATE_DIM),
      R_(OBS_DIM, OBS_DIM), H_(OBS_DIM, STATE_DIM),
      last_time_(0.0), last_update_time_(0.0), initialized_(false),
      prediction_horizon_(0.5), lost_count_(0), max_lost_count_(15) {
    
    // 初始化状态向量
    x_.setZero();
    
    // 初始化状态协方差矩阵
    P_.setIdentity();
    P_.block(3, 3, 3, 3) *= initial_velocity_std * initial_velocity_std;
    P_.block(6, 6, 3, 3) *= initial_acceleration_std * initial_acceleration_std;
    P_(9, 9) *= (M_PI / 4.0) * (M_PI / 4.0);  // 角度的初始不确定性 (45度)
    P_(10, 10) *= initial_angular_velocity_std * initial_angular_velocity_std;
    
    // 初始化过程噪声协方差矩阵
    Q_.setIdentity();
    Q_ *= process_noise_std * process_noise_std;
    Q_.block(3, 3, 3, 3) *= 2.0;
    Q_.block(6, 6, 3, 3) *= 5.0;
    Q_(9, 9) *= 3.0;    // 角度的过程噪声
    Q_(10, 10) *= 4.0;  // 角速度的过程噪声
    
    // 初始化测量噪声协方差矩阵
    R_.setIdentity();
    R_ *= measurement_noise_std * measurement_noise_std;
    
    // 初始化观测矩阵
    H_.setZero();
    H_(0, 0) = 1.0;  // x
    H_(1, 1) = 1.0;  // y
    H_(2, 2) = 1.0;  // z
}

double EnhancedEKF3D::normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void EnhancedEKF3D::initialize(const Eigen::Vector3d& measurement, double timestamp) {
    if (measurement.size() != 3) {
        throw std::invalid_argument("测量值必须是3维坐标 [x, y, z]");
    }
    
    // 初始化位置
    x_.segment(0, 3) = measurement;
    x_.segment(3, 3).setZero();  // 初始速度为0
    x_.segment(6, 3).setZero();  // 初始加速度为0
    x_(9) = 0.0;                 // 初始角度为0 (朝向Y轴正方向)
    x_(10) = 0.0;                // 初始角速度为0
    
    last_time_ = timestamp;
    last_update_time_ = 0.0;
    initialized_ = true;
    lost_count_ = 0;
    
    std::cout << "增强EKF初始化完成: 位置 [" << measurement(0) << ", " 
              << measurement(1) << ", " << measurement(2) << "]" << std::endl;
}

Eigen::VectorXd EnhancedEKF3D::predict(double timestamp) {
    if (!initialized_) {
        return x_;
    }
    
    // 计算时间差
    double dt = (last_time_ > 0) ? timestamp - last_time_ : 0.066;
    dt = std::max(0.001, std::min(dt, 0.2));
    
    // 获取当前状态
    double x = x_(0), y = x_(1), z = x_(2);
    double vx = x_(3), vy = x_(4), vz = x_(5);
    double ax = x_(6), ay = x_(7), az = x_(8);
    double theta = x_(9), omega = x_(10);
    
    // 更新角度
    double new_theta = normalize_angle(theta + omega * dt);
    
    // 基于角速度更新速度方向
    double current_speed = std::sqrt(vx * vx + vy * vy);
    if (current_speed > 0.3) {  // 只有在有显著运动时才应用角速度
        // 更新水平面速度分量
        double new_vx = current_speed * std::sin(new_theta);
        double new_vy = current_speed * std::cos(new_theta);
        
        // 平滑过渡，避免突变
        double alpha = 0.3;
        vx = alpha * new_vx + (1.0 - alpha) * vx;
        vy = alpha * new_vy + (1.0 - alpha) * vy;
    }
    
    // 状态转移矩阵 F
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    
    // 位置更新
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    F(0, 6) = 0.5 * dt * dt;
    F(1, 7) = 0.5 * dt * dt;
    F(2, 8) = 0.5 * dt * dt;
    
    // 速度更新
    F(3, 6) = dt;
    F(4, 7) = dt;
    F(5, 8) = dt;
    
    // 角度更新
    F(9, 10) = dt;
    
    // 手动更新状态（因为速度方向的更新是非线性的）
    x_(0) = x + vx * dt + 0.5 * ax * dt * dt;
    x_(1) = y + vy * dt + 0.5 * ay * dt * dt;
    x_(2) = z + vz * dt + 0.5 * az * dt * dt;
    x_(3) = vx + ax * dt;
    x_(4) = vy + ay * dt;
    x_(5) = vz + az * dt;
    // x_(6-8) 加速度保持不变
    x_(9) = new_theta;
    // x_(10) 角速度保持不变
    
    // 预测协方差
    P_ = F * P_ * F.transpose() + Q_;
    
    last_time_ = timestamp;
    
    return x_;
}

Eigen::VectorXd EnhancedEKF3D::update(const Eigen::Vector3d& measurement) {
    if (!initialized_) {
        throw std::runtime_error("滤波器未初始化，请先调用initialize()");
    }
    
    if (measurement.size() != 3) {
        throw std::invalid_argument("测量值必须是3维坐标 [x, y, z]");
    }
    
    Eigen::Vector3d y = measurement - H_ * x_;
    
    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
    
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    
    x_ = x_ + K * y;
    
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
    P_ = (I - K * H_) * P_;
    
    double vx = x_(3), vy = x_(4);
    double current_speed = std::sqrt(vx * vx + vy * vy);
    
    if (current_speed > 0.2) {  // 只有在有显著运动时才更新角度和角速度
        double measured_theta = std::atan2(vx, vy);
        double previous_theta = x_(9);
        
        double angle_diff = normalize_angle(measured_theta - previous_theta);
        
        if (last_update_time_ > 0) {
            double dt = std::max(0.001, std::min(0.2, last_time_ - last_update_time_));
            double estimated_omega = angle_diff / dt;
            double alpha = 0.3;
            x_(10) = alpha * estimated_omega + (1.0 - alpha) * x_(10);
        }
        
        x_(9) = normalize_angle(previous_theta + 0.4 * angle_diff);
    }
    
    last_update_time_ = last_time_;
    
    lost_count_ = 0;
    
    return x_;
}

std::tuple<double, double, double> EnhancedEKF3D::predict_future_position(std::optional<double> time_ahead) {
    if (!initialized_) {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
    
    double dt = time_ahead.value_or(prediction_horizon_);
    
    double x = x_(0), y = x_(1), z = x_(2);
    double vx = x_(3), vy = x_(4), vz = x_(5);
    double ax = x_(6), ay = x_(7), az = x_(8);
    double theta = x_(9), omega = x_(10);

    double current_speed = std::sqrt(vx * vx + vy * vy);
    double future_x, future_y, future_z;
    
    if (current_speed > 0.1 && std::abs(omega) > 0.01) {
        double radius = current_speed / std::abs(omega);
        double angle_change = omega * dt;
        double dx = radius * (std::sin(theta + angle_change) - std::sin(theta));
        double dy = radius * (std::cos(theta) - std::cos(theta + angle_change));
        
        future_x = x + dx + 0.5 * ax * dt * dt;
        future_y = y + dy + 0.5 * ay * dt * dt;
    } else {
        future_x = x + vx * dt + 0.5 * ax * dt * dt;
        future_y = y + vy * dt + 0.5 * ay * dt * dt;
    }
    
    future_z = z + vz * dt + 0.5 * az * dt * dt;
    
    return std::make_tuple(future_x, future_y, future_z);
}

std::tuple<double, double, double> EnhancedEKF3D::get_current_position() {
    if (!initialized_) {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
    
    return std::make_tuple(x_(0), x_(1), x_(2));
}

std::tuple<double, double, double> EnhancedEKF3D::get_current_velocity() {
    if (!initialized_) {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
    
    return std::make_tuple(x_(3), x_(4), x_(5));
}

std::tuple<double, double, double> EnhancedEKF3D::get_current_acceleration() {
    if (!initialized_) {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
    
    return std::make_tuple(x_(6), x_(7), x_(8));
}

std::optional<std::tuple<double, double, double>> EnhancedEKF3D::handle_lost_target(double timestamp) {
    if (!initialized_) {
        return std::nullopt;
    }
    
    lost_count_++;
    
    if (lost_count_ > max_lost_count_) {
        std::cout << "目标丢失时间过长，停止预测" << std::endl;
        return std::nullopt;
    }
    
    // 仅进行预测步骤
    predict(timestamp);
    
    // 增加过程噪声
    P_ *= 1.1;
    
    return get_current_position();
}

void EnhancedEKF3D::reset() {
    x_.setZero();
    P_.setIdentity();
    P_.block(3, 3, 3, 3) *= 0.5 * 0.5;
    P_.block(6, 6, 3, 3) *= 0.2 * 0.2;
    P_(9, 9) *= (M_PI / 4.0) * (M_PI / 4.0);
    P_(10, 10) *= 0.3 * 0.3;
    last_time_ = 0.0;
    last_update_time_ = 0.0;
    initialized_ = false;
    lost_count_ = 0;
    std::cout << "增强EKF已重置" << std::endl;
}

double EnhancedEKF3D::get_position_uncertainty() {
    if (!initialized_) {
        return std::numeric_limits<double>::infinity();
    }
    
    return P_.block(0, 0, 3, 3).trace();
}

void EnhancedEKF3D::set_process_noise(double std) {
    Q_.setIdentity();
    Q_ *= std * std;
    Q_.block(3, 3, 3, 3) *= 2.0;
    Q_.block(6, 6, 3, 3) *= 5.0;
    Q_(9, 9) *= 3.0;
    Q_(10, 10) *= 4.0;
}

void EnhancedEKF3D::set_measurement_noise(double std) {
    R_.setIdentity();
    R_ *= std * std;
}

}  // namespace track