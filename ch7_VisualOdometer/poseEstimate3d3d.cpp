//
// Created by zavier on 10/20/21.
//
// Use the SVD provided by Eigen to solve this problem,
// then use bundle adjustment to optimize result
//

#include <utility>

#include "visual_odometer.h"


static void SVDSolver(const std::vector<cv::Point3f>& pts1,
                      const std::vector<cv::Point3f>& pts2,
                      cv::Mat& R, cv::Mat& t);

static void bundleAdjustment(const std::vector<cv::Point3f>& pts1,
                             const std::vector<cv::Point3f>& pts2,
                             cv::Mat& R, cv::Mat& t, const cv::Mat& K);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    explicit EdgeProjectXYZRGBDPoseOnly( Eigen::Vector3d  point ) : _point(std::move(point)) {}

    void computeError() override
    {
        const auto* pose = dynamic_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    void linearizeOplus() override
    {
        auto* pose = dynamic_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }
    // read & write, remain empty
    bool read(std::istream& in) override {}
    bool write(std::ostream& out) const override {}
protected:
    Eigen::Vector3d _point;
};


void poseEstimate3d3d(std::vector<cv::KeyPoint> keypoint_1,
                      std::vector<cv::KeyPoint> keypoint_2,
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t, const cv::Mat& K,
                      const cv::Mat& depth_img_1,
                      const cv::Mat& depth_img_2,
                      bool check)
{
    std::vector<cv::Point3f> pts1, pts2;
    for (cv::DMatch m : matches) {
        ushort depth_1 = depth_img_1.ptr<unsigned short>(int(keypoint_1[m.queryIdx].pt.y))[int(keypoint_1[m.queryIdx].pt.x)];
        ushort depth_2 = depth_img_2.ptr<unsigned short>(int(keypoint_2[m.trainIdx].pt.y))[int(keypoint_2[m.trainIdx].pt.x)];
        if (depth_1 == 0 || depth_2 == 0)   // bad depth
            continue;
        cv::Point2d p1 = pixel2cam(keypoint_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoint_2[m.trainIdx].pt, K);
        double dd1 = double(depth_1) /5000.0;
        double dd2 = double(depth_2) /5000.0;
        pts1.emplace_back(p1.x*dd1, p1.y*dd1, dd1);
        pts2.emplace_back(p2.x*dd2, p2.y*dd2, dd2);
    }

    std::cout << "3d-3d pairs: " << pts1.size()  << std::endl;
    SVDSolver(pts1, pts2, R, t);
    std::cout << "ICP via SVD results: " << std::endl;
    std::cout << "R: " << R << std::endl;
    std::cout << "t: " << t << std::endl;
    std::cout << "R_inv: " << R.t()  << std::endl;
    std::cout << "t_inv: " << -R.t() *t << std::endl;

    std::cout << "Calling bundle adjustment ..." << std::endl;
    bundleAdjustment(pts1, pts2, R, t, K);

    if (check) {
        // check the front 5 points
        for (int i = 0; i < 5; ++i) {
            std::cout << "p1: " << pts1[i] << std::endl;
            std::cout << "p2: " << pts2[i] << std::endl;
            std::cout << "R * p2 + t: "
                << R * (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, pts2[i].z) + t << std::endl;
            std::cout << std::endl;
        }
    }
}

static void SVDSolver(const std::vector<cv::Point3f>& pts1,
                      const std::vector<cv::Point3f>& pts2,
                      cv::Mat& R, cv::Mat& t)
{
    cv::Point3f p1, p2;     // center of mass
    int N = int(pts1.size());
    for (int i = 0; i < N; ++i) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; ++i) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // Compute q1 * q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i)
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    std::cout << "W: " << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        for (int x = 0; x < 3; ++x)
            U(x, 2) *= -1;
    }

    std::cout << "U: " << U << std::endl;
    std::cout << "V: " << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // Convert to cv::Mat
    R = (cv::Mat_<double>(3,3) <<
        R_(0,0), R_(0,1), R_(0,2),
        R_(1,0), R_(1,1), R_(1,2),
        R_(2,0), R_(2,1), R_(2,2)
    );
    t = (cv::Mat_<double>(3,1) << t_(0,0), t_(1,0), t_(2,0));
}

static void bundleAdjustment(const std::vector<cv::Point3f>& pts1,
                             const std::vector<cv::Point3f>& pts2,
                             cv::Mat& R, cv::Mat& t, const cv::Mat& K)
{
    // g2o initialization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;  // pose: 6-dim, landmark: 3-dim
    auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<Block::PoseMatrixType>>(); // new g2o version
    auto block_solver = g2o::make_unique<Block>(std::move(linear_solver));
    auto* solver = new g2o::OptimizationAlgorithmGaussNewton(std::move(block_solver));
    g2o::SparseOptimizer optimizer;     // graph model
    optimizer.setAlgorithm(solver);

    // vertex
    auto* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d(0,0,0)
    ));
    optimizer.addVertex(pose);

    // edges
    int index = 1;
    for (size_t i = 0; i < pts1.size(); ++i) {
        auto* edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
        optimizer.addEdge(edge);
        index++;
    }

    // solving problem
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(end_time-start_time);

    // print details
    std::cout << "Time cost: " << time_cost.count() << std::endl;
    std::cout << "After optimization ... " << std::endl;
    std::cout << "T: " << Eigen::Isometry3d(pose->estimate()).matrix() << std::endl;
}


