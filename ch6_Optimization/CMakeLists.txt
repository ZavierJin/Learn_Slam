cmake_minimum_required(VERSION 3.20)
project(test1)

set(CMAKE_CXX_STANDARD 14)

add_executable(test1 main.cpp)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# matplotlib
find_package(Python2 COMPONENTS Development NumPy) # python
include_directories(${PYTHON_INCLUDE_DIRS}) # another way
# target_include_directories(myproject PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
include_directories("/home/zavier/Software/matplotlib-cpp/")

# link lib
target_link_libraries(test1 ${OpenCV_LIBS} ${CERES_LIBRARIES} Python2::Python Python2::NumPy)