//
// Created by zavier on 10/18/21.
//
#include "visual_odometer.h"

void triangulation(const std::vector<cv::KeyPoint>& keypoint_1,
                   const std::vector<cv::KeyPoint>& keypoint_2,
                   const std::vector<cv::DMatch>& matches,
                   const cv::Mat& R, const cv::Mat& t,
                   const cv::Mat& K, std::vector<cv::Point3d>& points)
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0
    );
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    // Convert pixel coordinates to camera coordinates
    std::vector<cv::Point2f> pts_1, pts_2;
    for (cv::DMatch m : matches) {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // Convert to non-homogeneous coordinates
    for (int i = 0; i < pts_4d.cols; ++i) {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // normalization
        cv::Point3d p(
            x.at<float>(0,0),
            x.at<float>(1,0),
            x.at<float>(2,0)
        );
        points.push_back(p);
    }
}

