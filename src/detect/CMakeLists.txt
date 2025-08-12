cmake_minimum_required(VERSION 3.8)
project(detect)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80 86 90)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)


# -------------- OpenCV --------------
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- libtorch --------------
# 手动设置CUDA路径（替换为你的实际路径）
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

# 显式声明nvToolsExt库
find_library(NVToolsExt_LIB libnvToolsExt.so PATHS "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib" REQUIRED)
add_library(CUDA::nvToolsExt SHARED IMPORTED)
set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION ${NVToolsExt_LIB})

list(APPEND CMAKE_PREFIX_PATH "/home/jw/libtorch")
set(Torch_DIR "/home/jw/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

# 设置 rpath 以便在运行时找到 PyTorch 库
set(CMAKE_INSTALL_RPATH "/home/jw/libtorch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Include directories
include_directories(include)

# Create ament index resource which references the libraries in the binary dir
set(node_plugins "")

# Add library for detect_node
add_library(detect_node_component SHARED
  src/detect_node.cpp
  src/detector.cpp
  src/reid.cpp
)

target_compile_definitions(detect_node_component PRIVATE "COMPOSITION_BUILDING_DLL")

ament_target_dependencies(detect_node_component
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(detect_node_component ${TORCH_LIBRARIES} ${OpenCV_LIBS})

rclcpp_components_register_nodes(detect_node_component "detect::Detect")
set(node_plugins "${node_plugins}detect::Detect;$<TARGET_FILE:detect_node_component>\n")

# Install targets
install(TARGETS
  detect_node_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(detect_node_component)
ament_export_dependencies(
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs
)

ament_package()