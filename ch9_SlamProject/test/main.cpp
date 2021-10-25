//
// Created by zavier on 10/23/21.
//

// -------------- test the visual odometer -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "my_slam/config.h"
#include "my_slam/visual_odometer_1.h"

int main ()
{
    my_slam::Config::setParameterFile(CONFIG_PATH);
    my_slam::VisualOdometer::Ptr vo_agent(new my_slam::VisualOdometer);

    std::string dataset_dir = my_slam::Config::get<std::string> ("dataset_dir");
    std::cout<<"dataset: "<<dataset_dir<<std::endl;
    std::ifstream fin ( dataset_dir+"/associate.txt" );
    if (!fin)
    {
        std::cout<<"Please generate the associate file called associate.txt!"<<std::endl;
        return 1;
    }

    std::vector<std::string> rgb_files, depth_files;
    std::vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        std::string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }

    my_slam::Camera::Ptr camera ( new my_slam::Camera );

    // visualization
    cv::viz::Viz3d vis("Visual Odometer");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Camera", camera_coor );

    std::cout<<"read total "<<rgb_files.size() <<" entries"<<std::endl;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cv::Mat color = cv::imread ( rgb_files[i] );
        cv::Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        my_slam::Frame::Ptr pFrame = my_slam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo_agent->addFrame ( pFrame );  // TODO: ERROR
        std::cout<<"VO costs time: "<<timer.elapsed()<<std::endl;

        if ( vo_agent->state_ == my_slam::VisualOdometer::LOST )
            break;
        Sophus::SE3d Tcw = pFrame->T_c_w_.inverse();

        // show the map and the camera pose 
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Tcw.matrix()(0,0), Tcw.matrix()(0,1), Tcw.matrix()(0,2),
                Tcw.matrix()(1,0), Tcw.matrix()(1,1), Tcw.matrix()(1,2),
                Tcw.matrix()(2,0), Tcw.matrix()(2,1), Tcw.matrix()(2,2)
            ),
            cv::Affine3d::Vec3(
                Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
            )
        );

        cv::imshow("image", color );
        cv::waitKey(1);
        vis.setWidgetPose( "Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}
