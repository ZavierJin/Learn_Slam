cmake_minimum_required(VERSION 3.20)
project(test1)

set(CMAKE_CXX_STANDARD 14)

add_executable(test1 main.cpp featureMatch.cpp visual_odometer.h poseEstimate2d2d.cpp utils.cpp triangulation.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# g2o
list(APPEND CMAKE_MODULE_PATH /home/zavier/Software/g2o/cmake_modules)
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# matplotlib cpp version
find_package(Python2 COMPONENTS Development NumPy)  # python
include_directories(${PYTHON_INCLUDE_DIRS})         # another way
# target_include_directories(myproject PRIVATE ${Python2_INCLUDE_DIRS} ${Python2_NumPy_INCLUDE_DIRS})
include_directories("/home/zavier/Software/matplotlib-cpp/")

# link lib
target_link_libraries(test1
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    g2o_core g2o_stuff
    Python2::Python Python2::NumPy
)