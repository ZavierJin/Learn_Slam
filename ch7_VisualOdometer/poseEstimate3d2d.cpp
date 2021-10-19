//
// Created by zavier on 10/19/21.
//
// Using bundle adjustment
//

#include "visual_odometer.h"


static void bundleAdjustment(const std::vector<cv::Point3f>& points_3d,
                             const std::vector<cv::Point2f>& points_2d,
                             cv::Mat& R, cv::Mat& t, const cv::Mat& K);


void poseEstimate3d2d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K,
                      const cv::Mat& depth_img_1, bool check)
{
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for (cv::DMatch m : matches) {
        ushort depth = depth_img_1.ptr<unsigned short>(int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        if (depth == 0)   // bad depth
            continue;
        float dd = depth / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        pts_3d.emplace_back(p1.x*dd, p1.y*dd, dd);      // what is emplace_back??
        pts_2d.push_back(keypoint_2[m.trainIdx].pt);
    }
    std::cout << "3D-2D pairs: " << pts_3d.size() <<std::endl;

    // Call the PNP solution of OpenCV, and select EPNP, DLS and other methods
    // R_mat is the form of rotation vector, which is transformed into matrix by Rodrigues formula
    cv::Mat R_vec;
    solvePnP(pts_3d, pts_2d, K, cv::Mat(), R_vec, t, false);
    cv::Rodrigues(R_vec, R);

    std::cout << "R: " << std::endl << R << std::endl;
    std::cout << "t: " << std::endl << t << std::endl;

    std::cout << "Calling bundle adjustment ... " << std::endl;
    bundleAdjustment(pts_3d, pts_2d, R, t, K);
}

void bundleAdjustment(const std::vector<cv::Point3f>& points_3d,
                      const std::vector<cv::Point2f>& points_2d,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K)
{
    // g2o initialization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;  // pose: 6-dim, landmark: 3-dim
    auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>(); // new g2o version
    auto block_solver = g2o::make_unique<Block>(std::move(linear_solver));
    auto* solver = new g2o::OptimizationAlgorithmLevenberg (std::move(block_solver));
    g2o::SparseOptimizer optimizer;     // graph model
    optimizer.setAlgorithm(solver);

    // vertex
    auto* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
        R.at<double> (0,0), R.at<double> (0,1), R.at<double> (0,2),
        R.at<double> (1,0), R.at<double> (1,1), R.at<double> (1,2),
        R.at<double> (2,0), R.at<double> (2,1), R.at<double> (2,2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
        R_mat,
        Eigen::Vector3d (t.at<double> (0,0), t.at<double> (1,0), t.at<double> (2,0))
    ));
    optimizer.addVertex(pose);

    int index = 1;
    for (const cv::Point3f& p : points_3d) {   // landmarks
        auto* point = new g2o::VertexPointXYZ();
        point->setId (index++);
        point->setEstimate (Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true); // see detail in ch11
        optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    auto* camera = new g2o::CameraParameters(
        K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2),
        K.at<double>(1,2)), 0
    );
    camera->setId(0);
    optimizer.addParameter(camera);

    // edges
    index = 1;
    for (const cv::Point2f& p : points_2d) {
        auto* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    // solving problem
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time);

    // print details
    std::cout << "Time cost: " << time_cost.count() << std::endl;
    std::cout << "After optimization ... " << std::endl;
    std::cout << "T: " << Eigen::Isometry3d(pose->estimate()).matrix() << std::endl;
}
