//
// Created by zavier on 10/23/21.
//

#include "my_slam/map.h"

namespace my_slam
{
void Map::insertKeyFrame ( const Frame::Ptr& frame )
{
    std::cout << "Key frame size = " << keyframes_.size() << std::endl;
    if (keyframes_.find(frame->id_) == keyframes_.end())
        keyframes_.insert( make_pair(frame->id_, frame) );
    else
        keyframes_[ frame->id_ ] = frame;
}

void Map::insertMapPoint (const MapPoint::Ptr& map_point)
{
    if (map_points_.find(map_point->id_) == map_points_.end())
        map_points_.insert( make_pair(map_point->id_, map_point));
    else
        map_points_[map_point->id_] = map_point;

}

}