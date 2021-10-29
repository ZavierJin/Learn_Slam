//
// Created by zavier on 10/28/21.
//

#ifndef MY_SLAM_G2O_TYPE_H
#define MY_SLAM_G2O_TYPE_H

#include "my_slam/common_include.h"
#include "camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/core/robust_kernel.h>
//#include <g2o/core/robust_kernel_impl.h>

namespace my_slam
{
class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void computeError() override;
    void linearizeOplus() override;
    bool read( std::istream& in ) override{}
    bool write( std::ostream& out) const override {}
};

// only to optimize the pose, no point
class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // TODO: not finish
    // Error: measure = R*point+t
    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read( std::istream& in ){}
    virtual bool write( std::ostream& out) const {}

    Eigen::Vector3d point_;
};

class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void computeError() override;
    void linearizeOplus() override;

    bool read( std::istream& in ) override{}
    bool write(std::ostream& os) const override {};

    Eigen::Vector3d point_;
    Camera* camera_;
};

}

/*
#include "my_slam/common_include.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace my_slam {
// vertex and edges used in g2o BA
// Pose vertex
class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

    // left multiplication on Sophus::SE3d
    void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }
};

// Landmark vertex
class VertexXYZ: public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void setToOriginImpl() override { _estimate = Eigen::Vector3d::Zero(); }

    void oplusImpl(const double *update) override {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }
};

// Only the univariate edges of the pose are estimated
class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectionPoseOnly(Eigen::Vector3d pos, Eigen::Matrix3d K)
            : _pos3d(std::move(pos)), _K(std::move(K)) {}

    void computeError() override {
        const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    void linearizeOplus() override {
        const VertexPose *v = dynamic_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                -fy * X * Zinv;
    }
    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

// Binary edge with map and pose
class EdgeProjection
        : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexPose, VertexXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Pass in camera internal and external parameters during construction
    EdgeProjection(Eigen::Matrix3d K, const Sophus::SE3d &cam_ext) : _K(std::move(K)) {
        _cam_ext = cam_ext;
    }

    void computeError() override {
        const VertexPose *v0 = dynamic_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ *v1 = dynamic_cast<VertexXYZ *>(_vertices[1]);
        Sophus::SE3d T = v0->estimate();
        Eigen::Vector3d pos_pixel = _K * (_cam_ext * (T * v1->estimate()));
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    void linearizeOplus() override {
        const VertexPose *v0 = dynamic_cast<VertexPose *>(_vertices[0]);
        const VertexXYZ *v1 = dynamic_cast<VertexXYZ *>(_vertices[1]);
        Sophus::SE3d T = v0->estimate();
        Eigen::Vector3d pw = v1->estimate();
        Eigen::Vector3d pos_cam = _cam_ext * T * pw;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
                -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
                fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
                -fy * X * Zinv;

        _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) *
                           _cam_ext.rotationMatrix() * T.rotationMatrix();
    }

    bool read(std::istream &in) override { return true; }
    bool write(std::ostream &out) const override { return true; }

private:
    Eigen::Matrix3d _K;
    Sophus::SE3d _cam_ext;
};

}  // namespace my_slam
*/
#endif //MY_SLAM_G2O_TYPE_H
