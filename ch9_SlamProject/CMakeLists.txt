cmake_minimum_required(VERSION 3.20)
project(my_slam)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH /home/zavier/Software/g2o/cmake_modules)
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
# G2O
find_package(G2O REQUIRED)
include_directories(
    ${G2O_INCLUDE_DIRS}
    "/usr/include/suitesparse"
)
# matplotlib cpp
find_package(Python2 COMPONENTS Development NumPy)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    "/home/zavier/Software/matplotlib-cpp/"
)

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    Sophus::Sophus
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d cxsparse
    Python2::Python Python2::NumPy
)

############### source and test ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
#add_executable(my_slam src/main.cpp)
#target_link_libraries( my_slam
#        ${THIRD_PARTY_LIBS}
#        )
