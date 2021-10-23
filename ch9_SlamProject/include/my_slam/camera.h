//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_CAMERA_H
#define MY_SLAM_CAMERA_H

#include "my_slam/common_include.h"
namespace my_slam
{
// Pinhole RGB-D camera model
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    float fx_{}, fy_{}, cx_{}, cy_{}, depth_scale_{}; // Camera intrinsics

public:
    Camera() = default;
    Camera(float fx, float fy, float cx, float cy, float depth_scale=0):
        fx_(fx), fy_(fy), cx_(cx), cy_(cy), depth_scale_(depth_scale) {}

    // coordinate transform: world, camera, pixel
    Eigen::Vector3d world2camera(const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w);
    Eigen::Vector3d camera2world(const Eigen::Vector3d& p_c, const Sophus::SE3d& T_c_w);
    Eigen::Vector2d camera2pixel(const Eigen::Vector3d& p_c);
    Eigen::Vector3d pixel2camera(const Eigen::Vector2d& p_p, double depth=1);
    Eigen::Vector3d pixel2world(const Eigen::Vector2d& p_p, const Sophus::SE3d& T_c_w, double depth=1);
    Eigen::Vector2d world2pixel(const Eigen::Vector3d& p_w, const Sophus::SE3d& T_c_w);
};
}

#endif //MY_SLAM_CAMERA_H
