/* Visual Odometer */

#include "visual_odometer.h"

int main()
{
    bool verify_2d2d = false;
    bool verify_triangulation = false;
    bool verify_3d2d = false;
    bool verify_3d3d = false;

//    featureExtraction();
    cv::Mat img_1 = cv::imread(IMG_PATH_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(IMG_PATH_2, CV_LOAD_IMAGE_COLOR);
    // The depth map is a 16-bit unsigned number, single channel image
    cv::Mat depth_img_1 = cv::imread(DEPTH_IMG_PATH_1, CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depth_img_2 = cv::imread(DEPTH_IMG_PATH_2, CV_LOAD_IMAGE_UNCHANGED);

    //-- Set camera internal parameters, TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    cv::Mat R,t;
    std::vector<cv::Point3d> points;

    //-- Feature matching
    std::cout << "============== Info ==============" << std::endl << std::endl;
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    std::vector<cv::DMatch> matches;
    featureMatch(img_1, img_2, keypoint_1, keypoint_2, matches);
    std::cout << "Matches count: " << matches.size() << std::endl;

    //-- 2D-2D
    // Estimating motion between two images
    std::cout << std::endl << "============== 2D-2D ==============" << std::endl << std::endl;
    poseEstimate2d2d(keypoint_1, keypoint_2, matches, R, t, K, verify_2d2d);
    triangulation(keypoint_1, keypoint_2, matches, R, t, K, points, verify_triangulation);

    //-- 3D-2D
    std::cout << std::endl << "============== 3D-2D ==============" << std::endl << std::endl;
    poseEstimate3d2d(keypoint_1, keypoint_2, matches, R, t, K, depth_img_1, verify_3d2d);

    //-- 3D-3D
    std::cout << std::endl << "============== 3D-3D ==============" << std::endl << std::endl;
    poseEstimate3d3d(keypoint_1, keypoint_2, matches, R, t, K, depth_img_1, depth_img_2, verify_3d3d);

    return 0;
}