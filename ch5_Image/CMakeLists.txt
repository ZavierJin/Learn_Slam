cmake_minimum_required(VERSION 3.20)
project(test1)

set(CMAKE_CXX_STANDARD 14)

add_executable(test1 main.cpp)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")

# PCL
find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# link lib
target_link_libraries(test1 ${OpenCV_LIBS} ${PCL_LIBRARIES})