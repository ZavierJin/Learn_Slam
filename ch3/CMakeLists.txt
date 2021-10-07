cmake_minimum_required(VERSION 3.20)
project(test1)

set(CMAKE_CXX_STANDARD 14)

add_executable(test1 main.cpp)

# Add Eigen
# Another way is use "find_package"
include_directories("/usr/include/eigen3")