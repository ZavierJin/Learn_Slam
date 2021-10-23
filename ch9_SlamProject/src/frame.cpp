//
// Created by zavier on 10/23/21.
//
#include "my_slam/frame.h"

#include <memory>
namespace my_slam
{

Frame::Ptr Frame::createFrame()
{
    static long factory_id = 0;
    return std::make_shared<Frame>(factory_id++ );
}

double Frame::findDepth(const cv::KeyPoint& kp)
{
    int x = cvRound(kp.pt.x);
    int y = cvRound(kp.pt.y);
    ushort d = depth_.ptr<ushort>(y)[x];
    if (d != 0) {
        return double(d) / camera_->depth_scale_;
    } else {
        // check the nearby points
        int dx[4] = {-1,0,1,0};
        int dy[4] = {0,-1,0,1};
        for (int i = 0; i < 4; ++i) {
            d = depth_.ptr<ushort>(y+dy[i])[x+dx[i]];
            if (d != 0)
                return double(d)/camera_->depth_scale_;
        }
    }
    return -1.0;
}

Eigen::Vector3d Frame::getCamCenter() const
{
    return T_c_w_.inverse().translation();
}

bool Frame::isInFrame(const Eigen::Vector3d& pt_world)
{
    Eigen::Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
    if (p_cam(2,0) < 0)
        return false;
    Eigen::Vector2d pixel = camera_->world2pixel(pt_world, T_c_w_);
    return pixel(0,0) > 0 && pixel(1,0) > 0
        && pixel(0,0) < color_.cols
        && pixel(1,0) < color_.rows;
}

}
