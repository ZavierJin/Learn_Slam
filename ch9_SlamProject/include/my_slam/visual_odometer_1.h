//
// Created by zavier on 10/25/21.
//

#ifndef MY_SLAM_VISUAL_ODOMETER_1_H
#define MY_SLAM_VISUAL_ODOMETER_1_H

#include "my_slam/common_include.h"
#include "my_slam/map.h"
#include <opencv2/features2d/features2d.hpp>

namespace my_slam
{
class VisualOdometer
{
public:
    typedef std::shared_ptr<VisualOdometer> Ptr;
    enum VOState {
        INITIALIZING = -1,
        OK = 0,
        LOST
    };

    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    Frame::Ptr  ref_;       // reference frame
    Frame::Ptr  curr_;      // current frame

    cv::Ptr<cv::ORB>                orb_;               // orb detector and computer
    std::vector<cv::Point3f>        pts_3d_ref_;        // 3D points in reference frame
    std::vector<cv::KeyPoint>       keypoint_curr_;     // keypoint in current frame
    std::vector<cv::KeyPoint>       keypoint_ref_;      // keypoint in reference frame, for debug
    cv::Mat                         descriptor_curr_;   // descriptor in current frame
    cv::Mat                         descriptor_ref_;    // descriptor in reference frame
    std::vector<cv::DMatch>         feature_matches_;
    std::vector<int>                error_depth_index;

    Sophus::SE3d    T_c_r_estimated_;       // the estimated pose of current frame
    int             num_inliers_;           // number of inlier features in icp
    int             num_lost_;              // number of lost times

    int     num_of_features_;       // number of features
    double  scale_factor_;          // scale in image pyramid
    int     level_pyramid_;         // number of pyramid levels
    float   match_ratio_;           // ratio for selecting  good matches
    int     max_num_lost_;          // max number of continuous lost times
    int     min_inliers_;           // minimum inlier

    double  key_frame_min_rot;      // minimal rotation of two key-frames
    double  key_frame_min_trans;    // minimal translation of two key-frames

public:
    VisualOdometer();
    ~VisualOdometer() = default;

    bool addFrame( const Frame::Ptr& frame );      // add a new frame

protected:
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();
    void featureMatching();
    void poseEstimationPnP();
    void setRef3DPoints();

    void addKeyFrame();
    bool checkEstimatedPose();
    bool checkKeyFrame();
};

}

#endif //MY_SLAM_VISUAL_ODOMETER_1_H
