//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_COMMON_INCLUDE_H
#define MY_SLAM_COMMON_INCLUDE_H

// define the commonly included file to avoid a long include list
// std
#include <iostream>
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <utility>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
//using Eigen::Vector2d;
//using Eigen::Vector3d;

//// for Sophus
#include <sophus/se3.hpp>
//using Sophus::SE3;

// opencv
#include <opencv2/core/core.hpp>
//using cv::Mat;

#define CONFIG_PATH "../config/default.yaml"

#endif //MY_SLAM_COMMON_INCLUDE_H
