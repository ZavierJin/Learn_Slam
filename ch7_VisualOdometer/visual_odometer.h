//
// Created by zavier on 10/17/21.
//

#ifndef VISUAL_ODOMETER_H
#define VISUAL_ODOMETER_H

#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_sba.h>

#define IMG_PATH_1 "1.png"
#define IMG_PATH_2 "2.png"
#define DEPTH_IMG_PATH_1 "1_depth.png"

__attribute__((unused)) void featureExtraction();

void featureMatch(const cv::Mat& img_1, const cv::Mat& img_2,
                  std::vector<cv::KeyPoint>& keypoint_1,
                  std::vector<cv::KeyPoint>& keypoint_2,
                  std::vector<cv::DMatch>& matches);

void poseEstimate2d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K,
                      bool check);

void triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                   const std::vector<cv::KeyPoint>& keypoint_2,
                   const std::vector<cv::DMatch>& matches,
                   const cv::Mat& R, const cv::Mat& t, const cv::Mat& K,
                   std::vector<cv::Point3d>& points, bool check);

void poseEstimate3d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K,
                      const cv::Mat& depth_img_1, bool check);

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);


#endif // VISUAL_ODOMETER_H
