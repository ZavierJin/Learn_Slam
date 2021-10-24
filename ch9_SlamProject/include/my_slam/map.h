//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_MAP_H
#define MY_SLAM_MAP_H

#include "my_slam/common_include.h"
#include "my_slam/frame.h"
#include "my_slam/map_point.h"

namespace my_slam
{
class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;
    std::unordered_map<unsigned long, MapPoint::Ptr>  map_points_;        // all landmarks
    std::unordered_map<unsigned long, Frame::Ptr>     keyframes_;         // all key-frames

    Map() = default;
    void insertKeyFrame(const Frame::Ptr& frame);
    void insertMapPoint(const MapPoint::Ptr& map_point);
};

}

#endif //MY_SLAM_MAP_H
