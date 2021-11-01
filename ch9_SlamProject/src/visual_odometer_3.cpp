//
// Created by zavier on 11/1/21.
//


#include "my_slam/visual_odometer_3.h"
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
        map_(new Map), num_lost_(0), num_inliers_(0),
        matcher_flann_(new cv::flann::LshIndexParams(5,10,2))
{
    num_of_features_        = Config::get<int>("number_of_features");
    scale_factor_           = Config::get<double>("scale_factor");
    level_pyramid_          = Config::get<int>("level_pyramid");
    match_ratio_            = Config::get<float>("match_ratio");
    max_num_lost_           = Config::get<int>("max_num_lost");
    min_inliers_            = Config::get<int>("min_inliers");
    key_frame_min_rot       = Config::get<double>("keyframe_rotation");
    key_frame_min_trans     = Config::get<double>("keyframe_translation");
    map_point_erase_ratio_  = Config::get<double>("map_point_erase_ratio");
    orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
}

bool VisualOdometer::addFrame(const Frame::Ptr& frame)
{
    std::cout << "----------------------------------" << std::endl;
    switch (state_) {
        case INITIALIZING:
            state_ = TRACKING;
            ref_ = curr_ = frame;
            K_ = (cv::Mat_<double>(3,3) <<
                curr_->camera_->fx_, 0, curr_->camera_->cx_,
                0, curr_->camera_->fy_, curr_->camera_->cy_,
                0, 0, 1
            );
            extractKeyPoints();
            computeDescriptors();
            addKeyFrame();      // the first frame is a key-frame
            std::cout << "Finish initialization." << std::endl;
            break;
        case TRACKING:
            curr_ = frame;
            curr_->T_c_w_ = ref_->T_c_w_;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if (checkEstimatedPose()) {     // a good estimation
                std::cout << "A good estimation." << std::endl;
                curr_->T_c_w_ = T_c_w_estimated_;
//                optimizeMap();    // TODO: Optimize map size will cause an error, iter_ptr++ become null_ptr
                num_lost_ = 0;
                if (checkKeyFrame()) // is a key-frame
                    addKeyFrame();
            } else {    // bad estimation due to various reasons
                std::cout << "Bad estimation due to various reasons." << std::endl;
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

void VisualOdometer::featureMatching()
{
    std::vector<cv::DMatch> matches;
    // select the candidates in map
    cv::Mat desp_map;
    std::vector<MapPoint::Ptr> candidate;
    for (auto& all_points: map_->map_points_) {
        MapPoint::Ptr& p = all_points.second;
        // check if p in curr frame image
        if (curr_->isInFrame(p->pos_)) {
            // add to candidate
            p->visible_times_++;
            candidate.push_back(p);
            desp_map.push_back(p->descriptor_);
        }
    }

    matcher_flann_.match(desp_map, descriptor_curr_, matches );

    //-- Filter the matching point pairs
    double min_dist = 10000;
    // Find out the minimum distance and maximum distance between all matches
    // The distance between the most similar and the least similar two groups of points
    for (int i = 0; i < descriptor_curr_.rows; ++i) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
    }

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    double threshold = 30.0;
    for (auto& match : matches) {
        if (match.distance <= cv::max(2 * min_dist, threshold)) {
            match_3dpts_.push_back(candidate[match.queryIdx]);
            match_2dkp_index_.push_back(match.trainIdx);
        }
    }
}

void VisualOdometer::poseEstimationPnP()
{
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (int index : match_2dkp_index_)
        pts_2d.push_back(keypoint_curr_[index].pt);
    for (const MapPoint::Ptr& pt : match_3dpts_)
        pts_3d.push_back(pt->getPositionCV());

    // Call the PNP solution of OpenCV, and select EPNP, DLS and other methods
    // R_mat is the form of rotation vector, which is transformed into matrix by Rodrigues formula
    cv::Mat R_vec, t_vec, inliers;
    cv::solvePnPRansac(pts_3d, pts_2d, K_, cv::Mat(),R_vec,
                       t_vec, false, 100, 4.0, 0.99, inliers );
//        solvePnP(pts_3d, pts_2d, K_, cv::Mat(), R_vec, t_vec, false);
    num_inliers_ = inliers.rows;
    std::cout << "PNP inliers number: " << num_inliers_ << "/" << pts_3d.size() << std::endl;

    cv::Mat R_mat_cv;
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t;
    cv::Rodrigues(R_vec, R_mat_cv);
    cv::cv2eigen(R_mat_cv, R_mat);
    t = Eigen::Vector3d(t_vec.at<double>(0,0), t_vec.at<double>(1,0), t_vec.at<double>(2,0));
//        std::cout << "R_vec: " << std::endl << R_vec << std::endl;
//        std::cout << "CV-R: " << std::endl << R_mat_cv << std::endl;
//    std::cout << "Eigen-R: " << std::endl << R_mat << std::endl;
//    std::cout << "t_vec: " << std::endl << t_vec << std::endl;
    T_c_w_estimated_ = Sophus::SE3d(R_mat, t);

    auto* pose = new g2o::VertexSE3Expmap; // camera pose
    bundleAdjustment(pts_3d, pts_2d, pose, inliers);

    T_c_w_estimated_ = Sophus::SE3d(
            Eigen::Isometry3d(pose->estimate()).rotation(),
            Eigen::Isometry3d(pose->estimate()).translation()
    );
}

void VisualOdometer::bundleAdjustment(const std::vector<cv::Point3f>& points_3d,
                                      const std::vector<cv::Point2f>& points_2d,
                                      g2o::VertexSE3Expmap* pose, cv::Mat& inliers)
{
    // g2o initialization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;  // pose: 6-dim, landmark: 2-dim
    auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>(); // new g2o version
    auto block_solver = g2o::make_unique<Block>(std::move(linear_solver));
    auto* solver = new g2o::OptimizationAlgorithmLevenberg (std::move(block_solver));
    g2o::SparseOptimizer optimizer;     // graph model
    optimizer.setAlgorithm(solver);

    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(T_c_w_estimated_.rotationMatrix(), T_c_w_estimated_.translation()));
    optimizer.addVertex(pose);

    // edges
    for (int i = 0; i < inliers.rows; ++i) {
        int index = inliers.at<int>(i,0);
        // 3D -> 2D projection
        auto * edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId(i);
        edge->setVertex(0, pose);
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Eigen::Vector3d(points_3d[index].x, points_3d[index].y, points_3d[index].z );
        edge->setMeasurement( Eigen::Vector2d(points_2d[index].x, points_2d[index].y) );
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);
}

void VisualOdometer::addKeyFrame()
{
    if (map_->keyframes_.empty()) {
        // first key-frame, add all 3d points into map
        for (size_t i = 0; i < keypoint_curr_.size(); ++i) {
            double d = curr_->findDepth(keypoint_curr_[i]);
            if ( d < 0 )
                continue;
            Eigen::Vector3d p_world = ref_->camera_->pixel2world (
                Eigen::Vector2d ( keypoint_curr_[i].pt.x, keypoint_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Eigen::Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptor_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}

bool VisualOdometer::checkKeyFrame()
{
    Sophus::SE3d T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Eigen::Vector3d trans = d.head<3>();
    Eigen::Vector3d rot = d.tail<3>();
    if (rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans)
        return true;
    return false;
}

void VisualOdometer::addMapPoints()
{
    // add the new map points into map
    std::vector<bool> matched(keypoint_curr_.size(), false);
    for (int index : match_2dkp_index_)
        matched[index] = true;
    for (int i = 0; i < keypoint_curr_.size(); ++i) {
        if (matched[i])
            continue;
        double d = ref_->findDepth ( keypoint_curr_[i] );
        if (d < 0)
            continue;
        Eigen::Vector3d p_world = ref_->camera_->pixel2world (
                Eigen::Vector2d ( keypoint_curr_[i].pt.x, keypoint_curr_[i].pt.y ),
                curr_->T_c_w_, d
        );
        Eigen::Vector3d camera_center = p_world - ref_->getCamCenter();
        camera_center.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, camera_center, descriptor_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

// Optimize map size will cause an error, iter_ptr++ become null_ptr
void VisualOdometer::optimizeMap()
{
    // remove the hardly seen and no visible points 
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); ++iter) {
        if (!curr_->isInFrame(iter->second->pos_)) {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_) / float(iter->second->visible_times_);
        if ( match_ratio < map_point_erase_ratio_ ) {
            iter = map_->map_points_.erase(iter);
            continue;
        }

        double angle = getViewAngle(curr_, iter->second);
        if ( angle > M_PI/6. ) {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if (!iter->second->good_) {
            // TODO try triangulate this map point 
        }
    }

    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 ) {
        // TODO map is too large, remove some one
        map_point_erase_ratio_ += 0.05;
    } else
        map_point_erase_ratio_ = 0.1;
    std::cout << "Map points: " << map_->map_points_.size() << std::endl;
}

double VisualOdometer::getViewAngle(const Frame::Ptr& frame, const MapPoint::Ptr& point)
{
    Eigen::Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

bool VisualOdometer::checkEstimatedPose()
{
    // Check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        std::cout << "Reject because inlier is too small: " << num_inliers_ << std::endl;
        return false;
    }
    // If the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_w_estimated_.log();
    if (d.norm() > 5.0){
        std::cout << "Reject because motion is too large: " << d.norm() << std::endl;
        return false;
    }
    return true;
}

}

