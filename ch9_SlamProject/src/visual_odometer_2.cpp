//
// Created by zavier on 10/27/21.
//


#include "my_slam/visual_odometer_2.h"
#include "my_slam/config.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <boost/timer.hpp>


namespace my_slam
{

VisualOdometer::VisualOdometer():
    state_(INITIALIZING), ref_(nullptr), curr_(nullptr),
    map_(new Map), num_lost_(0), num_inliers_(0)
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<int> ( "max_num_lost" );          // modified from float to int
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_ );
}

bool VisualOdometer::addFrame(const Frame::Ptr& frame)
{
    switch (state_) {
        case INITIALIZING:
            state_ = TRACKING;
            curr_ = frame;
            map_->insertKeyFrame ( frame );
            K_ = ( cv::Mat_<double>(3,3) <<
                curr_->camera_->fx_, 0, curr_->camera_->cx_,
                0, curr_->camera_->fy_, curr_->camera_->cy_,
                0, 0, 1
            );
            extractKeyPoints();
            computeDescriptors();
            stateUpdate();
            break;
        case TRACKING:
            curr_ = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if (checkEstimatedPose()) {     // a good estimation
//                std::cout << "A good estimation." << std::endl;
                curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w
                stateUpdate();
                num_lost_ = 0;
                if (checkKeyFrame()) // is a key-frame
                    addKeyFrame();
            } else {    // bad estimation due to various reasons
//                std::cout << "Bad estimation due to various reasons." << std::endl;
                num_lost_++;
                if (num_lost_ > max_num_lost_)
                    state_ = LOST;
                return false;
            }
            break;
        case LOST:
            std::cout << "VO has lost." << std::endl;
            break;
    }
    return true;
}

void VisualOdometer::extractKeyPoints()
{
    orb_->detect(curr_->color_, keypoint_curr_);
}

void VisualOdometer::computeDescriptors()
{
    orb_->compute(curr_->color_, keypoint_curr_, descriptor_curr_);
}

void VisualOdometer::stateUpdate()
{
    ref_ = curr_;
    keypoint_ref_.clear();
    keypoint_ref_ = keypoint_curr_;
    descriptor_ref_ = cv::Mat();        // Reset descriptor!!!!
    for (size_t i = 0; i < keypoint_curr_.size(); ++i) {
        descriptor_ref_.push_back(descriptor_curr_.row(int(i)));
    }
}

void VisualOdometer::featureMatching()
{
    std::vector<cv::DMatch> matches;
//        cv::BFMatcher matcher ( cv::NORM_HAMMING );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptor_ref_, descriptor_curr_, matches );
//        // select the best matches
//        float min_dist = std::min_element (
//                matches.begin(), matches.end(),
//                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
//                {
//                    return m1.distance < m2.distance;
//                } )->distance;
    //-- Filter the matching point pairs
    double min_dist = 10000, max_dist = 0;
    // Find out the minimum distance and maximum distance between all matches
    // The distance between the most similar and the least similar two groups of points
    for (int i = 0; i < descriptor_ref_.rows; ++i) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    feature_matches_.clear();
    double threshold = 30.0;
    for (auto & match : matches) {
        if (match.distance <= cv::max(2 * min_dist, threshold))
            feature_matches_.push_back(match);
    }
//        //-- Draw matching result
//        cv::Mat img_good_match;
//        cv::drawMatches(ref_->color_, keypoint_ref_, curr_->color_, keypoint_curr_, feature_matches_, img_good_match);
//        cv::imshow("Good Matches", img_good_match);
//        cv::waitKey(0);
}

void VisualOdometer::poseEstimationPnP()
{
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : feature_matches_) {
        // mathod # 1
        double d = ref_->findDepth(keypoint_ref_[m.queryIdx]);
        if ( d > 0) {
            Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(
                Eigen::Vector2d(keypoint_ref_[m.queryIdx].pt.x, keypoint_ref_[m.queryIdx].pt.y), d
            );
            pts_3d.emplace_back( p_cam(0,0), p_cam(1,0), p_cam(2,0) );
            pts_2d.push_back(keypoint_curr_[m.trainIdx].pt);
//                descriptor_ref_.push_back(descriptor_curr_.row(i));
        }
    }
    // Call the PNP solution of OpenCV, and select EPNP, DLS and other methods
    // R_mat is the form of rotation vector, which is transformed into matrix by Rodrigues formula
    cv::Mat R_vec, t_vec, inliers;
    cv::solvePnPRansac(pts_3d, pts_2d, K_, cv::Mat(), R_vec, t_vec, false, 100, 4.0, 0.99, inliers );
//        solvePnP(pts_3d, pts_2d, K_, cv::Mat(), R_vec, t_vec, false);
    num_inliers_ = inliers.rows;
    std::cout << "PNP inliers number: " << num_inliers_ << "/" << pts_3d.size() << std::endl;

    cv::Mat R_mat_cv;
    Eigen::Matrix3d R_mat;
    cv::Rodrigues(R_vec, R_mat_cv);
    cv::cv2eigen(R_mat_cv, R_mat);
//        std::cout << "R_vec: " << std::endl << R_vec << std::endl;
//        std::cout << "CV-R: " << std::endl << R_mat_cv << std::endl;
//    std::cout << "Eigen-R: " << std::endl << R_mat << std::endl;
//    std::cout << "t_vec: " << std::endl << t_vec << std::endl;
    T_c_r_estimated_ = Sophus::SE3d(
            R_mat,
            // Sophus::SO3d(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
            Eigen::Vector3d( t_vec.at<double>(0,0), t_vec.at<double>(1,0), t_vec.at<double>(2,0))
    );
//        std::cout << "Calling bundle adjustment ... " << std::endl;
//        bundleAdjustment(pts_3d, pts_2d, R, t, K);
}

bool VisualOdometer::checkEstimatedPose()
{
    // Check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        std::cout << "reject because inlier is too small: " << num_inliers_ << std::endl;
        return false;
    }
    // If the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > 5.0 )
    {
        std::cout << "reject because motion is too large: " << d.norm() << std::endl;
        return false;
    }
    return true;
}

bool VisualOdometer::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Eigen::Vector3d trans = d.head<3>();
    Eigen::Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometer::addKeyFrame()
{
    std::cout << "adding a key-frame" << std::endl;
    map_->insertKeyFrame ( curr_ );
}

}