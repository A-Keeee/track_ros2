## track_ros2

本仓库包含多个ROS2功能包，使用OAK深度相机实现YOLO11n-pose/ReID/EKF进行单目标识别和追踪。

### 目录结构

- `src/`：源代码目录，包含各功能包：
  - `detect/` 目标检测相关包
  - `juwei_launch/` 启动文件包
  - `oak_d_camera/` 相机驱动与节点
  - `track/` 跟踪相关包
- `build/`：构建输出目录
- `install/`：安装输出目录
- `log/`：构建和运行日志

### 依赖环境

- ROS2（Humble）
- colcon构建工具
- C++/Python依赖（根据各功能包需求）
- Libtorch(注意适配CUDA版本)
- OpenCV
- depthai

### Nvidia Jetson安装特别说明

1. **安装Torch**
    
    请参考[Nvidia论坛](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)，根据Jetpack版本安装对应的Pytorch版本。

2. **环境搭建**

    安装后，将 `/torch` 路径添加到 `detect`、`track` 等包的 `CMakeLists.txt` 中的 `TORCH_DIR` 变量。


### 快速开始

1. **安装依赖**
   
   请确保已正确安装ROS2和colcon。可参考[ROS2官方文档](https://docs.ros.org/)。

2. **构建工作空间**
   ```bash
   cd ～/track_ros2
   source /opt/ros/humble/setup.bash
   colcon build
   ```

3. **环境配置**
   ```bash
   source install/setup.bash
   ```

4. **运行示例**
   
   ```
   以`juwei_launch`包启动：
   ```bash
   ros2 launch juwei_launch track_system.launch.py
   ```

### 贡献

欢迎提交issue和PR，完善功能与文档。

### 许可证

本项目遵循MIT License。
