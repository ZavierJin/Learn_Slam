//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_MAP_POINT_H
#define MY_SLAM_MAP_POINT_H

#include "my_slam/common_include.h"

namespace my_slam
{

class Frame;
class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long       id_;                // ID
    Eigen::Vector3d     pos_;               // Position in world
    Eigen::Vector3d     norm_;              // Normal of viewing direction
    cv::Mat             descriptor_;        // Descriptor for matching
    int                 observed_times_;    // being observed by feature matching algo.
    int                 correct_times_;     // being an in-liner in pose estimation

public:
    MapPoint(): id_(-1), pos_(Eigen::Vector3d(0,0,0)), norm_(Eigen::Vector3d(0,0,0)),
        observed_times_(0), correct_times_(0) {}
    MapPoint( long id, Eigen::Vector3d position, Eigen::Vector3d norm ):
        id_(id), pos_(std::move(position)), norm_(std::move(norm)), observed_times_(0), correct_times_(0) {}

    // factory function
    static MapPoint::Ptr createMapPoint();
};

}

#endif //MY_SLAM_MAP_POINT_H
