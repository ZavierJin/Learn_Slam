//
// Created by zavier on 10/17/21.
//
#include "visual_odometer.h"

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return {
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    };
}
