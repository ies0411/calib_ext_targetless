cmake_minimum_required(VERSION 3.0.2)
project(calib-extrinsic-targetless)

# # Compile as C++14, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

# SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# # Find catkin macros and libraries
# # if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
# # is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

# # System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV)
# find_package(Open3D)
find_package(Threads)
# find_package(FMT REQUIRED)
find_package(Ceres REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs
)

# link_directories(${Open3D_LIBRARY_DIRS})

include_directories(

  # include
  INCLUDE_DIRS include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # ${Open3D_INCLUDE_DIRS}
)

add_executable(lidar_camera_calib src/lidar_camera_calib.cpp include/lidar_camera_calib.hpp include/optimize.hpp)

# add_executable(lidar_camera_multi_calib src/lidar_camera_multi_calib.cpp include/lidar_camera_calib.hpp)
# add_executable(bag_to_pcd src/bag_to_pcd.cpp)
target_link_libraries(lidar_camera_calib
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
)

# target_link_libraries(lidar_camera_multi_calib
# ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
# )

# target_link_libraries(bag_to_pcd
# ${catkin_LIBRARIES}  ${PCL_LIBRARIES}
# )