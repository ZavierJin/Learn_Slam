//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_FRAME_H
#define MY_SLAM_FRAME_H

#include <utility>

#include "my_slam/common_include.h"
#include "my_slam/camera.h"

namespace my_slam
{
// forward declare
class MapPoint;

class Frame
{
public:
    typedef std::shared_ptr<Frame>  Ptr;
    unsigned long                   id_;         // id of this frame
    double                          time_stamp_; // when it is recorded
    Sophus::SE3d                    T_c_w_;      // transform from world to camera
    Camera::Ptr                     camera_;     // Pinhole RGB-D Camera model
    cv::Mat                         color_, depth_; // color and depth image

public: // data members
    Frame(): id_(-1), time_stamp_(-1), camera_(nullptr) {}
    explicit Frame(long id, double time_stamp=0, const Sophus::SE3d& T_c_w=Sophus::SE3d(),
                   Camera::Ptr camera=nullptr, cv::Mat color=cv::Mat(), cv::Mat depth=cv::Mat()):
                   id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(std::move(camera)),
                   color_(std::move(color)), depth_(std::move(depth)) {};
    ~Frame() = default;

    // factory function
    static Frame::Ptr createFrame();

    // find the depth in depth map
    double findDepth( const cv::KeyPoint& kp );

    // Get Camera Center
    Eigen::Vector3d getCamCenter() const;

    // check if a point is in this frame
    bool isInFrame( const Eigen::Vector3d& pt_world );
};

}

#endif //MY_SLAM_FRAME_H
