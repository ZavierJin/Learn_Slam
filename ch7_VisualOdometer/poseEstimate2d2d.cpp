//
// Created by zavier on 10/17/21.
//
#include "visual_odometer.h"

void poseEstimate2d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t)
{
    //-- Set camera internal parameters, TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- Convert the matching point to the form of vector<Point2f>
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (auto & match : matches) {
        points1.push_back(keypoint_1[match.queryIdx].pt);
        points2.push_back(keypoint_2[match.trainIdx].pt);
    }

    //-- Compute fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "Fundamental matrix: " << std::endl << fundamental_matrix << std::endl;

    //-- Compute essential matrix
    cv::Point2d principal_point(325.1, 249.7);	// Camera optical center, tum dataset calibration value
    double focal_length = 521;			// Camera focal length, tum dataset calibration value
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
