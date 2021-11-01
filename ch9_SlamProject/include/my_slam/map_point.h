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
    unsigned long      id_;        // ID
    static unsigned long factory_id_;    // factory id
    bool        good_;      // wheter a good point
    Eigen::Vector3d     pos_;               // Position in world
    Eigen::Vector3d     norm_;              // Normal of viewing direction
    cv::Mat             descriptor_;        // Descriptor for matching

    std::list<Frame*>    observed_frames_;   // key-frames that can observe this point

    int         matched_times_;     // being an inliner in pose estimation
    int         visible_times_;     // being visible in current frame

public:
    MapPoint(): id_(-1), pos_(Eigen::Vector3d(0,0,0)), norm_(Eigen::Vector3d(0,0,0)),
            good_(true), visible_times_(0), matched_times_(0) {};
    MapPoint(unsigned long id, Eigen::Vector3d position, Eigen::Vector3d norm,
            Frame* frame=nullptr, cv::Mat descriptor=cv::Mat());

    inline cv::Point3f getPositionCV() const {
        return {
            static_cast<float>(pos_(0,0)),
            static_cast<float>(pos_(1,0)),
            static_cast<float>(pos_(2,0))
        };
    }

    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint(
            const Eigen::Vector3d& pos_world,
            const Eigen::Vector3d& norm_,
            const cv::Mat& descriptor,
            Frame* frame );
};

}

#endif //MY_SLAM_MAP_POINT_H
