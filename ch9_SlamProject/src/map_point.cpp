//
// Created by zavier on 10/23/21.
//
#include "my_slam/map_point.h"

namespace my_slam
{

MapPoint::Ptr MapPoint::createMapPoint()
{
    static long factory_id = 0;
    return std::make_shared<MapPoint>(
        factory_id++, Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)
    );
}

}
