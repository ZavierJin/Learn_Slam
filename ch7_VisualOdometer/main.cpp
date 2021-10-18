/* Visual Odometer */

#include "visual_odometer.h"

int main()
{
    bool verify_2d2d = true;
    bool verify_triangulation = false;

//    featureExtraction();
    cv::Mat img_1 = cv::imread(IMG_PATH_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(IMG_PATH_2, CV_LOAD_IMAGE_COLOR);

    //-- Set camera internal parameters, TUM Freiburg2
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    std::vector<cv::DMatch> matches;
    featureMatch(img_1, img_2, keypoint_1, keypoint_2, matches);
    std::cout << "Matches count: " << matches.size() << std::endl;

    //-- Estimating motion between two images
    cv::Mat R,t;
    poseEstimate2d2d(keypoint_1, keypoint_2, matches, R, t, K);

    //-- Triangulation
    std::vector<cv::Point3d> points;
    triangulation(keypoint_1, keypoint_2, matches, R, t, K, points);

    //-- Verify pose estimation 2D-2D
    if (verify_2d2d) {
        //-- Verify E = t^R*scale
        cv::Mat t_x = (cv::Mat_<double>(3, 3) <<
                                              0, -t.at<double>(2, 0), t.at<double>(1, 0),
                t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                -t.at<double>(1, 0), t.at<double>(0, 0), 0);

        std::cout << "t^R=" << std::endl << t_x * R << std::endl;

        //-- Verify polar constraints
        for (cv::DMatch m: matches) {
            cv::Point2d pt1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
            cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
            cv::Point2d pt2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
            cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
            cv::Mat d = y2.t() * t_x * R * y1;
            std::cout << "Polar constraint: " << d << std::endl;
        }
    }

    //-- Verify the projection relationship between triangulated points and feature points
    if (verify_triangulation) {
        for (int i = 0; i < matches.size(); ++i) {
            cv::Point2d pt1_cam = pixel2cam(keypoint_1[matches[i].queryIdx].pt, K);
            cv::Point2d pt1_cam_3d(
                    points[i].x / points[i].z,
                    points[i].y / points[i].z
            );
            std::cout << "Point in the first camera frame: " << pt1_cam << std::endl;
            std::cout << "Point projected from 3D " << pt1_cam_3d << ", d=" << points[i].z << std::endl;

            // second image
            cv::Point2f pt2_cam = pixel2cam(keypoint_2[matches[i].trainIdx].pt, K);
            cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
            pt2_trans /= pt2_trans.at<double>(2, 0);
            std::cout << "Point in the second camera frame: " << pt2_cam << std::endl;
            std::cout << "Point reprojected from second frame: " << pt2_trans.t() << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;
}