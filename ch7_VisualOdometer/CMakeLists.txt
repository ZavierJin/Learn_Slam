cmake_minimum_required(VERSION 3.20)
project(test1)

set(CMAKE_CXX_STANDARD 14)

add_executable(test1
    main.cpp
    featureMatch.cpp
    visual_odometer.h
    poseEstimate2d2d.cpp
    utils.cpp
    triangulation.cpp
    poseEstimate3d2d.cpp
    poseEstimate3d3d.cpp
)

list(APPEND CMAKE_MODULE_PATH /home/zavier/Software/g2o/cmake_modules)
find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Python2 COMPONENTS Development NumPy)

include_directories(
    "/usr/include/eigen3"
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    "/usr/include/suitesparse"
    ${PYTHON_INCLUDE_DIRS}
    "/home/zavier/Software/matplotlib-cpp/"
)

# SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

# link lib
target_link_libraries(test1
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d cxsparse
    Python2::Python Python2::NumPy
)