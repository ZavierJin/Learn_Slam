//
// Created by zavier on 10/28/21.
//

#include "my_slam/g2o_type.h"

namespace my_slam
{

void EdgeProjectXYZRGBD::computeError()
{
    const auto* point = dynamic_cast<const g2o::VertexPointXYZ*> ( _vertices[0] );
    const auto* pose = dynamic_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
    _error = _measurement - pose->estimate().map ( point->estimate() );
}

void EdgeProjectXYZRGBD::linearizeOplus()
{
    auto* pose = dynamic_cast<g2o::VertexSE3Expmap *> ( _vertices[1] );
    g2o::SE3Quat T ( pose->estimate() );
    auto* point = dynamic_cast<g2o::VertexPointXYZ*> ( _vertices[0] );
    Eigen::Vector3d xyz = point->estimate();
    Eigen::Vector3d xyz_trans = T.map ( xyz );
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];

    _jacobianOplusXi = - T.rotation().toRotationMatrix();

    _jacobianOplusXj ( 0,0 ) = 0;
    _jacobianOplusXj ( 0,1 ) = -z;
    _jacobianOplusXj ( 0,2 ) = y;
    _jacobianOplusXj ( 0,3 ) = -1;
    _jacobianOplusXj ( 0,4 ) = 0;
    _jacobianOplusXj ( 0,5 ) = 0;

    _jacobianOplusXj ( 1,0 ) = z;
    _jacobianOplusXj ( 1,1 ) = 0;
    _jacobianOplusXj ( 1,2 ) = -x;
    _jacobianOplusXj ( 1,3 ) = 0;
    _jacobianOplusXj ( 1,4 ) = -1;
    _jacobianOplusXj ( 1,5 ) = 0;

    _jacobianOplusXj ( 2,0 ) = -y;
    _jacobianOplusXj ( 2,1 ) = x;
    _jacobianOplusXj ( 2,2 ) = 0;
    _jacobianOplusXj ( 2,3 ) = 0;
    _jacobianOplusXj ( 2,4 ) = 0;
    _jacobianOplusXj ( 2,5 ) = -1;
}


void EdgeProjectXYZ2UVPoseOnly::computeError()
{
    const auto* pose = dynamic_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
    _error = _measurement - camera_->camera2pixel (
            pose->estimate().map(point_) );
}

void EdgeProjectXYZ2UVPoseOnly::linearizeOplus()
{
    auto* pose = dynamic_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
    g2o::SE3Quat T ( pose->estimate() );
    Eigen::Vector3d xyz_trans = T.map ( point_ );
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z*z;

    _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx_;
    _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx_;
    _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
    _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
    _jacobianOplusXi ( 0,4 ) = 0;
    _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;

    _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
    _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
    _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
    _jacobianOplusXi ( 1,3 ) = 0;
    _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
    _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;
}


}
