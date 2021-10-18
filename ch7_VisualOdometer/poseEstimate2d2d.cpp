//
// Created by zavier on 10/17/21.
//
#include "visual_odometer.h"

void poseEstimate2d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K)
{
    //-- Convert the matching point to the form of vector<Point2f>
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (auto& match : matches) {
        points1.push_back(keypoint_1[match.queryIdx].pt);
        points2.push_back(keypoint_2[match.trainIdx].pt);
    }

    //-- Compute fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "Fundamental matrix: " << std::endl << fundamental_matrix << std::endl;

    //-- Compute essential matrix
    // Camera optical center, tum dataset calibration value
    cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
    // Camera focal length, tum dataset calibration value
    double focal_length = K.at<double>(1, 1);
    cv::Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "Essential matrix: " << std::endl << essential_matrix << std::endl;

    //-- Compute homography matrix
    cv::Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "Homography matrix: " << std::endl << homography_matrix << std::endl;

    //-- Recover rotation and translation information from the essential matrix
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "t: " << std::endl << t << std::endl;
}
