//
// Created by zavier on 10/23/21.
//
#include "my_slam/map_point.h"

#include <memory>
#include <utility>

namespace my_slam
{
MapPoint::MapPoint(long unsigned int id, Eigen::Vector3d position,
                   Eigen::Vector3d norm, Frame* frame, cv::Mat descriptor):
       id_(id), pos_(std::move(position)), norm_(std::move(norm)), good_(true),
       visible_times_(1), matched_times_(1), descriptor_(std::move(descriptor))
{
    observed_frames_.push_back(frame);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
    return std::make_shared<MapPoint>(
        factory_id_++,
        Eigen::Vector3d(0,0,0),
        Eigen::Vector3d(0,0,0)
    );
}

MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d& pos_world, const Eigen::Vector3d& norm,
                                       const cv::Mat& descriptor, Frame* frame)
{
    return std::make_shared<MapPoint>(factory_id_++, pos_world, norm, frame, descriptor);
}

unsigned long MapPoint::factory_id_ = 0;

}
