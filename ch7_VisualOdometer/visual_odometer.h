//
// Created by zavier on 10/17/21.
//

#ifndef VISUAL_ODOMETER_H
#define VISUAL_ODOMETER_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define IMG_PATH_1 "1.png"
#define IMG_PATH_2 "2.png"

void featureExtraction();

void featureMatch(const cv::Mat& img_1, const cv::Mat& img_2,
                  std::vector<cv::KeyPoint>& keypoint_1,
                  std::vector<cv::KeyPoint>& keypoint_2,
                  std::vector<cv::DMatch>& matches);

void poseEstimate2d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K);

void triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                   const std::vector<cv::KeyPoint>& keypoint_2,
                   const std::vector<cv::DMatch>& matches,
                   const cv::Mat& R, const cv::Mat& t,
                   const cv::Mat& K, std::vector<cv::Point3d>& points);

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K);


#endif // VISUAL_ODOMETER_H
