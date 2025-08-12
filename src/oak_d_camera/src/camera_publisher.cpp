/*
MIT License

Copyright (c) 2024 José Miguel Guerrero Hernández

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cstdio>
#include <functional>
#include <iostream>
#include <tuple>

#include "camera_info_manager/camera_info_manager.hpp"
#include "rclcpp/executors.hpp"
#include "rclcpp/node.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

// Includes DepthAI libraries needed to work with the OAK-D device and pipeline
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/IMU.hpp"
#include "depthai/pipeline/node/MonoCamera.hpp"
#include "depthai/pipeline/node/StereoDepth.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/BridgePublisher.hpp"
#include "depthai_bridge/DisparityConverter.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_bridge/ImuConverter.hpp"
#include "depthai/pipeline/node/ColorCamera.hpp"

std::vector<std::string> usbStrings = {"UNKNOWN", "LOW", "FULL", "HIGH", "SUPER", "SUPER_PLUS"};

// Function to configure and create the DepthAI pipeline
std::tuple<dai::Pipeline, int, int, int, int> createPipeline(
  bool lrcheck, bool extended, bool subpixel, int confidence, int LRchecktresh, bool use_depth,
  bool use_disparity, bool use_lr_raw)
{
  // Creates the processing pipeline
  dai::Pipeline pipeline;

  // Disables chunk size for XLink transfer
  pipeline.setXLinkChunkSize(0);

  // Sets the resolution for the mono cameras and color camera
  dai::node::MonoCamera::Properties::SensorResolution monoResolution =
    dai::node::MonoCamera::Properties::SensorResolution::THE_400_P; // Changed to 480P for consistency
  dai::ColorCameraProperties::SensorResolution colorResolution =
    dai::ColorCameraProperties::SensorResolution::THE_800_P;
  int stereoWidth = 640, stereoHeight = 400, colorWidth = 640, colorHeight = 400; // Updated stereoHeight


  // ------------------------------
  // IMU
  // ------------------------------
  // Creates node for IMU and XLink output connection
  std::shared_ptr<dai::node::IMU> imu = pipeline.create<dai::node::IMU>();
  std::shared_ptr<dai::node::XLinkOut> xoutImu = pipeline.create<dai::node::XLinkOut>();
  xoutImu->setStreamName("imu");

  // Configures IMU node
  imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, 500);
  imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, 400);
  imu->setBatchReportThreshold(5);
  imu->setMaxBatchReports(20);

  // Links IMU to its XLink output
  imu->out.link(xoutImu->input);


  // ------------------------------
  // Color Camera
  // ------------------------------
  // Creates node for Color camera and XLink output connection
  std::shared_ptr<dai::node::ColorCamera> colorCam = pipeline.create<dai::node::ColorCamera>();
  std::shared_ptr<dai::node::XLinkOut> xoutColor = pipeline.create<dai::node::XLinkOut>();

  // Configures stream names for Color output
  xoutColor->setStreamName("color");

  // Color camera configuration
  colorCam->setBoardSocket(dai::CameraBoardSocket::CAM_A);
  colorCam->setResolution(colorResolution);
  colorCam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
  colorCam->setFps(35);

  // Link Color camera to its XLink output
  colorCam->video.link(xoutColor->input);


  // ------------------------------
  // Mono Cameras and Stereo Depth
  // ------------------------------
  std::shared_ptr<dai::node::MonoCamera> monoLeft, monoRight;
  std::shared_ptr<dai::node::XLinkOut> xoutLeft, xoutRight, xoutLeftRect, xoutRightRect;

  std::shared_ptr<dai::node::StereoDepth> stereo;
  std::shared_ptr<dai::node::XLinkOut> xoutDepth, xoutDepthDisp;

  if (use_depth || use_disparity || use_lr_raw) {
    // Creates nodes for left and right mono cameras and stereo depth
    monoLeft = pipeline.create<dai::node::MonoCamera>();
    monoRight = pipeline.create<dai::node::MonoCamera>();
    stereo = pipeline.create<dai::node::StereoDepth>();

    // Creates XLink output connections for left and right rectified images
    xoutLeftRect = pipeline.create<dai::node::XLinkOut>();
    xoutRightRect = pipeline.create<dai::node::XLinkOut>();

    // Configures stream names for output connections
    xoutLeftRect->setStreamName("left_rect");
    xoutRightRect->setStreamName("right_rect");

    // Mono camera configuration (left and right)
    monoLeft->setResolution(monoResolution);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    monoLeft->setFps(40);
    monoRight->setResolution(monoResolution);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);
    monoRight->setFps(40);

    // Stereo depth node configuration
    stereo->initialConfig.setConfidenceThreshold(confidence);
    stereo->setRectifyEdgeFillColor(0);
    stereo->initialConfig.setLeftRightCheckThreshold(LRchecktresh);
    stereo->setLeftRightCheck(lrcheck);
    stereo->setExtendedDisparity(extended);
    stereo->setSubpixel(subpixel);
    stereo->setDepthAlign(dai::CameraBoardSocket::CAM_A);

    // Link mono cameras to stereo depth node
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    // Link mono cameras to rectified outputs
    stereo->rectifiedLeft.link(xoutLeftRect->input);
    stereo->rectifiedRight.link(xoutRightRect->input);
  }

  // If using left and right raw images
  if (use_lr_raw) {
    xoutLeft = pipeline.create<dai::node::XLinkOut>();
    xoutRight = pipeline.create<dai::node::XLinkOut>();
    xoutLeft->setStreamName("left");
    xoutRight->setStreamName("right");
    stereo->syncedLeft.link(xoutLeft->input);
    stereo->syncedRight.link(xoutRight->input);
    std::cout << "Using left and right raw images" << std::endl;
  }

  // If using depth images
  if (use_depth) {
    xoutDepth = pipeline.create<dai::node::XLinkOut>();
    xoutDepth->setStreamName("depth");
    stereo->depth.link(xoutDepth->input);
    std::cout << "Using depth images" << std::endl;
  }

  // If using disparity images
  if (use_disparity) {
    xoutDepthDisp = pipeline.create<dai::node::XLinkOut>();
    xoutDepthDisp->setStreamName("disparity");
    stereo->disparity.link(xoutDepthDisp->input);
    std::cout << "Using disparity images" << std::endl;
  }

  // Returns the pipeline and configured dimensions
  return std::make_tuple(pipeline, stereoWidth, stereoHeight, colorWidth, colorHeight);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("camera_node");

  // Declare and get ROS parameters
  std::string tfPrefix;
  bool lrcheck, extended, subpixel, use_depth, use_disparity, use_lr_raw, pc_color, only_color,
    use_pointcloud;
  int confidence, LRchecktresh;
  int monoWidth, monoHeight, colorWidth, colorHeight;

  // Restored all parameters from the reference code
  node->declare_parameter("tf_prefix", "oak");
  node->declare_parameter("lrcheck", true);
  node->declare_parameter("extended", false);
  node->declare_parameter("subpixel", true);
  node->declare_parameter("confidence", 200);
  node->declare_parameter("LRchecktresh", 5);
  node->declare_parameter("use_depth", true);
  node->declare_parameter("use_disparity", false); // Defaulting to false, can be changed
  node->declare_parameter("use_lr_raw", false);    // Defaulting to false
  node->declare_parameter("pc_color", true);
  node->declare_parameter("only_color", false);
  node->declare_parameter("use_pointcloud", false); // Defaulting to false

  node->get_parameter("tf_prefix", tfPrefix);
  node->get_parameter("lrcheck", lrcheck);
  node->get_parameter("extended", extended);
  node->get_parameter("subpixel", subpixel);
  node->get_parameter("confidence", confidence);
  node->get_parameter("LRchecktresh", LRchecktresh);
  node->get_parameter("use_depth", use_depth);
  node->get_parameter("use_disparity", use_disparity);
  node->get_parameter("use_lr_raw", use_lr_raw);
  node->get_parameter("pc_color", pc_color);
  node->get_parameter("only_color", only_color);
  node->get_parameter("use_pointcloud", use_pointcloud);

  if (only_color) {
    use_depth = false;
    use_disparity = false;
    use_lr_raw = false;
    use_pointcloud = false;
  }

  // Creates the pipeline with specified parameters
  dai::Pipeline pipeline;
  std::tie(pipeline, monoWidth, monoHeight, colorWidth, colorHeight) = createPipeline(
        lrcheck, extended, subpixel, confidence, LRchecktresh, use_depth, use_disparity,
    use_lr_raw);

  // Initialize DepthAI devices with configured pipeline
  dai::Device device(pipeline);

  // Reads calibration data from the device
  auto calibrationHandler = device.readCalibration();

  // Show configuration
  RCLCPP_INFO(node->get_logger(), "-------------------------------");
  RCLCPP_INFO(node->get_logger(), "System Information:");
  RCLCPP_INFO(node->get_logger(), "- Device Name: %s", calibrationHandler.getEepromData().boardName.c_str());
  RCLCPP_INFO(node->get_logger(), "- Device MxID : %s", device.getMxId().c_str());
  RCLCPP_INFO(node->get_logger(), "- Device USB status: %s", usbStrings[static_cast<int32_t>(device.getUsbSpeed())].c_str());
  RCLCPP_INFO(node->get_logger(), "- Color resolution: %dx%d", colorWidth, colorHeight);
  if (!only_color) {
      RCLCPP_INFO(node->get_logger(), "- Mono resolution: %dx%d", monoWidth, monoHeight);
  }
  if (use_depth) RCLCPP_INFO(node->get_logger(), "- Depth images activated");
  if (use_disparity) RCLCPP_INFO(node->get_logger(), "- Disparity images activated");
  if (use_lr_raw) RCLCPP_INFO(node->get_logger(), "- Left and right raw images activated");
  RCLCPP_INFO(node->get_logger(), "-------------------------------");


  // Create publishers
  std::unique_ptr<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>
    colorPublish, leftPublish, rightPublish, leftRectPublish, rightRectPublish, depthPublish;
  std::unique_ptr<dai::rosBridge::BridgePublisher<stereo_msgs::msg::DisparityImage, dai::ImgFrame>> dispPublish;

  // Create converters
  dai::rosBridge::ImageConverter colorConverter(tfPrefix + "_rgb_camera_optical_frame", false);
  dai::rosBridge::ImageConverter leftConverter(tfPrefix + "_left_camera_optical_frame", true);
  dai::rosBridge::ImageConverter rightConverter(tfPrefix + "_right_camera_optical_frame", true);
  dai::rosBridge::DisparityConverter dispConverter(tfPrefix + "_right_camera_optical_frame", 880, 7.5, 20, 2000);

  // Create a specific converter for the depth image
  // The frame ID depends on whether we align to the color camera or the right mono camera
  dai::rosBridge::ImageConverter depthConverter(
    pc_color ? (tfPrefix + "_rgb_camera_optical_frame") : (tfPrefix + "_right_camera_optical_frame"), 
    true); // is_depth must be true

  // Get camera info
  auto colorCameraInfo = colorConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_A, colorWidth, colorHeight);
  auto leftCameraInfo = leftConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_B, monoWidth, monoHeight);
  auto rightCameraInfo = rightConverter.calibrationToCameraInfo(calibrationHandler, dai::CameraBoardSocket::CAM_C, monoWidth, monoHeight);
  auto depthCameraInfo = pc_color ? colorCameraInfo : rightCameraInfo;

  // Setup RGB publisher
  auto colorQueue = device.getOutputQueue("color", 30, false);
  colorPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
    colorQueue, node, "color/image",
    std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &colorConverter, std::placeholders::_1, std::placeholders::_2),
    30, colorCameraInfo, "color");
  colorPublish->addPublisherCallback();
  RCLCPP_INFO(node->get_logger(), "📸 Publishing color images to: /color/image");


  // --- CRITICAL FIX: Add publishers for rectified images to prevent pipeline stall ---
  if (use_depth || use_disparity || use_lr_raw) {
    auto leftRectQueue = device.getOutputQueue("left_rect", 30, false);
    leftRectPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
      leftRectQueue, node, "left_rect/image",
      std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &leftConverter, std::placeholders::_1, std::placeholders::_2),
      30, leftCameraInfo, "left_rect");
    leftRectPublish->addPublisherCallback();
    RCLCPP_INFO(node->get_logger(), " rectified left images to: /left_rect/image");

    auto rightRectQueue = device.getOutputQueue("right_rect", 30, false);
    rightRectPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
      rightRectQueue, node, "right_rect/image",
      std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rightConverter, std::placeholders::_1, std::placeholders::_2),
      30, rightCameraInfo, "right_rect");
    rightRectPublish->addPublisherCallback();
    RCLCPP_INFO(node->get_logger(), " rectified right images to: /right_rect/image");
  }

  // Setup Raw stereo publishers if enabled
  if (use_lr_raw) {
    auto leftQueue = device.getOutputQueue("left", 30, false);
    leftPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
      leftQueue, node, "left/image",
      std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &leftConverter, std::placeholders::_1, std::placeholders::_2),
      30, leftCameraInfo, "left");
    leftPublish->addPublisherCallback();

    auto rightQueue = device.getOutputQueue("right", 30, false);
    rightPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
      rightQueue, node, "right/image",
      std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &rightConverter, std::placeholders::_1, std::placeholders::_2),
      30, rightCameraInfo, "right");
    rightPublish->addPublisherCallback();
  }

  // Setup Depth publisher if enabled
  if (use_depth) {
    // --- CRITICAL FIX: Use non-blocking queue ---
    auto stereoQueue = device.getOutputQueue("depth", 30, false);
    depthPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Image, dai::ImgFrame>>(
      stereoQueue, node, "stereo/depth", // Changed topic to stereo/depth for consistency
      std::bind(&dai::rosBridge::ImageConverter::toRosMsg, &depthConverter, std::placeholders::_1, std::placeholders::_2),
      30, depthCameraInfo, "stereo");
    depthPublish->addPublisherCallback();
    RCLCPP_INFO(node->get_logger(), "🏔️ Publishing depth images to: /stereo/depth");
  }

  // Setup Disparity publisher if enabled
  if (use_disparity) {
    auto stereoQueueDisp = device.getOutputQueue("disparity", 30, false);
    dispPublish = std::make_unique<dai::rosBridge::BridgePublisher<stereo_msgs::msg::DisparityImage, dai::ImgFrame>>(
      stereoQueueDisp, node, "stereo/disparity",
      std::bind(&dai::rosBridge::DisparityConverter::toRosMsg, &dispConverter, std::placeholders::_1, std::placeholders::_2),
      30, rightCameraInfo, "stereo");
    dispPublish->addPublisherCallback();
    RCLCPP_INFO(node->get_logger(), "↔️ Publishing disparity images to: /stereo/disparity");
  }

  // Setup IMU publisher
  dai::ros::ImuSyncMethod imuMode = static_cast<dai::ros::ImuSyncMethod>(1); // COPY
  dai::rosBridge::ImuConverter imuConverter(tfPrefix + "_imu_frame", imuMode, 0.0, 0.0);
  auto imuQueue = device.getOutputQueue("imu", 30, false);
  auto imuPublish = std::make_unique<dai::rosBridge::BridgePublisher<sensor_msgs::msg::Imu, dai::IMUData>>(
      imuQueue, node, "imu/data",
      std::bind(&dai::rosBridge::ImuConverter::toRosMsg, &imuConverter, std::placeholders::_1, std::placeholders::_2),
      30, "", "imu");
  imuPublish->addPublisherCallback();
  RCLCPP_INFO(node->get_logger(), "🧭 Publishing IMU data to: /imu/data");

  // Spins the node
  RCLCPP_INFO(node->get_logger(), "🚀 Camera node started successfully!");
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}