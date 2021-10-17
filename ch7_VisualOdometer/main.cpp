/* Visual Odometer */

#include "visual_odometer.h"

int main()
{
//    featureExtraction();
    cv::Mat img_1 = cv::imread(IMG_PATH_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(IMG_PATH_2, CV_LOAD_IMAGE_COLOR);

    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    std::vector<cv::DMatch> matches;
    featureMatch(img_1, img_2, keypoint_1, keypoint_2, matches);
    std::cout << "Matches count: " << matches.size() << std::endl;

    //-- 估计两张图像间运动
    cv::Mat R,t;
    poseEstimate2d2d(keypoint_1, keypoint_2, matches, R, t);

    //-- 验证E=t^R*scale
    cv::Mat t_x = (cv::Mat_<double>(3, 3) <<
            0, -t.at<double> ( 2,0 ),   t.at<double> (1,0 ),
            t.at<double> ( 2,0 ),       0,                      -t.at<double> ( 0,0 ),
            -t.at<double> ( 1,0 ),      t.at<double> ( 0,0 ),      0 );

    std::cout << "t^R=" << std::endl << t_x*R << std::endl;

    //-- 验证对极约束
    cv::Mat K = (cv::Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m: matches) {
        cv::Point2d pt1 = pixel2cam ( keypoint_1[ m.queryIdx ].pt, K );
        cv::Mat y1 = (cv::Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        cv::Point2d pt2 = pixel2cam ( keypoint_2[ m.trainIdx ].pt, K );
        cv::Mat y2 = (cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "Epipolar constraint = " << d << std::endl;
    }
    return 0;
}