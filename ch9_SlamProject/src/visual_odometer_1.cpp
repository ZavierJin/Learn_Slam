//
// Created by zavier on 10/25/21.
//

#include "my_slam/config.h"
#include "my_slam/visual_odometer_1.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <boost/timer.hpp>



namespace my_slam
{

    VisualOdometer::VisualOdometer():
        state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ),
        map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
    {
        num_of_features_    = Config::get<int> ( "number_of_features" );
        scale_factor_       = Config::get<double> ( "scale_factor" );
        level_pyramid_      = Config::get<int> ( "level_pyramid" );
        match_ratio_        = Config::get<float> ( "match_ratio" );
        max_num_lost_       = Config::get<float> ( "max_num_lost" );          // modified from float to int
        min_inliers_        = Config::get<int> ( "min_inliers" );
        key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
        key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_ );
    }

    bool VisualOdometer::addFrame ( const Frame::Ptr& frame )
    {
        switch ( state_ )
        {
            case INITIALIZING:
            {
                state_ = OK;
                curr_ = ref_ = frame;
                map_->insertKeyFrame ( frame );
                // extract features from first frame 
                extractKeyPoints();
                computeDescriptors();
                // compute the 3d position of features in ref frame 
                setRef3DPoints();
                break;
            }
            case OK:
            {
                curr_ = frame;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                poseEstimationPnP();
                if ( checkEstimatedPose() == true ) // a good estimation
                {
                    curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
                    ref_ = curr_;
                    setRef3DPoints();
                    num_lost_ = 0;
                    if ( checkKeyFrame() == true ) // is a key-frame
                    {
                        addKeyFrame();
                    }
                }
                else // bad estimation due to various reasons
                {
                    num_lost_++;
                    if ( num_lost_ > max_num_lost_ )
                    {
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                std::cout<<"vo has lost."<<std::endl;
                break;
            }
        }

        return true;
    }

    void VisualOdometer::extractKeyPoints()
    {
        orb_->detect ( curr_->color_, keypoint_curr_ );
    }

    void VisualOdometer::computeDescriptors()
    {
        orb_->compute ( curr_->color_, keypoint_curr_, descriptor_curr_ );
    }

    void VisualOdometer::featureMatching()
    {
        // match desp_ref and desp_curr, use OpenCV's brute force match 
        std::vector<cv::DMatch> matches;
        cv::BFMatcher matcher ( cv::NORM_HAMMING );
        matcher.match ( descriptor_ref_, descriptor_curr_, matches );
        // select the best matches
        float min_dis = std::min_element (
                matches.begin(), matches.end(),
                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
                {
                    return m1.distance < m2.distance;
                } )->distance;

        feature_matches_.clear();
        for ( cv::DMatch& m : matches )
        {
            if ( m.distance < std::max<float> ( min_dis*match_ratio_, 30.0 ) )
            {
                feature_matches_.push_back(m);
            }
        }
        std::cout<<"good matches: "<<feature_matches_.size()<<std::endl;
    }

    void VisualOdometer::setRef3DPoints()
    {
        // select the features with depth measurements 
        pts_3d_ref_.clear();
        descriptor_ref_ = cv::Mat();
        for ( size_t i=0; i<keypoint_curr_.size(); i++ )
        {
            double d = ref_->findDepth(keypoint_curr_[i]);
            if ( d > 0)
            {
                Eigen::Vector3d p_cam = ref_->camera_->pixel2camera(
                        Eigen::Vector2d(keypoint_curr_[i].pt.x, keypoint_curr_[i].pt.y), d
                );
                pts_3d_ref_.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
                descriptor_ref_.push_back(descriptor_curr_.row(i));
            }
        }
    }

    void VisualOdometer::poseEstimationPnP()
    {
        // construct the 3d 2d observations
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;

        for ( cv::DMatch m:feature_matches_ )
        {
            pts3d.push_back( pts_3d_ref_[m.queryIdx] );
            pts2d.push_back( keypoint_curr_[m.trainIdx].pt );
        }

        cv::Mat K = ( cv::Mat_<double>(3,3)<<
                                       ref_->camera_->fx_, 0, ref_->camera_->cx_,
                0, ref_->camera_->fy_, ref_->camera_->cy_,
                0,0,1
        );
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        num_inliers_ = inliers.rows;
        std::cout<<"pnp inliers: "<<num_inliers_<<std::endl;
        Eigen::Matrix3d R_mat;
        cv::cv2eigen(rvec, R_mat);  // TODO: ERROR, R is not orthogonal
        T_c_r_estimated_ = Sophus::SE3d(
            R_mat,
            // Sophus::SO3d(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
            Eigen::Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
        );
    }

    bool VisualOdometer::checkEstimatedPose()
    {
        // check if the estimated pose is good
        if ( num_inliers_ < min_inliers_ )
        {
            std::cout<<"reject because inlier is too small: "<<num_inliers_<<std::endl;
            return false;
        }
        // if the motion is too large, it is probably wrong
        Sophus::Vector6d d = T_c_r_estimated_.log();
        if ( d.norm() > 5.0 )
        {
            std::cout<<"reject because motion is too large: "<<d.norm()<<std::endl;
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
        std::cout<<"adding a key-frame"<<std::endl;
        map_->insertKeyFrame ( curr_ );
    }

}