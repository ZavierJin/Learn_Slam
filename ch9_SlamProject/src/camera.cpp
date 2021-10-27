//
// Created by zavier on 10/23/21.
//

#include "my_slam/camera.h"
#include "my_slam/config.h"
namespace my_slam
{

Camera::Camera()
{
    fx_ = Config::get<float>("camera.fx");
    fy_ = Config::get<float>("camera.fy");
    cx_ = Config::get<float>("camera.cx");
    cy_ = Config::get<float>("camera.cy");
    depth_scale_ = Config::get<float>("camera.depth_scale");
}


Eigen::Vector3d my_slam::Camera::world2camera(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
    return T_c_w * p_w;
}

Eigen::Vector3d my_slam::Camera::camera2world(const Eigen::Vector3d &p_c, const Sophus::SE3d &T_c_w)
{
    return T_c_w.inverse() * p_c;
}

Eigen::Vector2d my_slam::Camera::camera2pixel(const Eigen::Vector3d &p_c)
{
    return {
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_
    };
}

Eigen::Vector3d my_slam::Camera::pixel2camera(const Eigen::Vector2d &p_p, double depth)
{
    return {
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth
    };
}

Eigen::Vector3d my_slam::Camera::pixel2world(const Eigen::Vector2d &p_p, const Sophus::SE3d &T_c_w, double depth)
{
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

Eigen::Vector2d my_slam::Camera::world2pixel(const Eigen::Vector3d &p_w, const Sophus::SE3d &T_c_w)
{
    return camera2pixel(world2camera(p_w, T_c_w));
}



}



