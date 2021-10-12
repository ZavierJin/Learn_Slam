#include <iostream>
#include <chrono> // timing
#include <fstream>
#include <boost/format.hpp> // format string

#include <opencv2/core/core.hpp> // version 2 ??
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#define IMAGE_PATH "ubuntu.png"
//#define IMAGE_PATH "/home/zavier/cpp_code/test1/ubuntu.png"
#define IMAGE_NUM 5
#define POSE_PATH "pose.txt"
#define CLOUD_SAVE_PATH "map.pcd"


void joinMap()
{
    std::vector<cv::Mat> color_img, depth_img;
    std::vector<Eigen::Isometry3d> cam_pose; // camera pose

    std::ifstream fin(POSE_PATH);
    if (!fin){
        std::cerr << "Pose path error!" << std::endl;
        return;
    }

    for (int i = 0; i < IMAGE_NUM; ++i){
        boost::format fmt("./%s/%d.%s");     // image file format
        color_img.push_back(cv::imread((fmt%"color"%(i+1)%"png").str()));
        depth_img.push_back(cv::imread((fmt%"depth"%(i+1)%"pgm").str(), -1));

        double data[7] = {0};
        for (auto& d:data)
            fin >> d;   // ????
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        cam_pose.push_back(T);
    }

    // camera instinct parameter
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depth_scale = 1000.0;

    std::cout << "Start joint point cloud ..." << std::endl;
    // define point cloud data format, use XYZRGB
    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // create a new point cloud
    PointCloud::Ptr point_cloud(new PointCloud);
    for (int i = 0; i < IMAGE_NUM; ++i){
        std::cout << "Start translate image ..." << std::endl;
        cv::Mat color = color_img[i];
        cv::Mat depth = depth_img[i];
        Eigen::Isometry3d T = cam_pose[i];
        for (int v = 0; v < color.rows; ++v)
            for (int u = 0; u < color.cols; ++u){
                unsigned int d = depth.ptr<unsigned short>(v)[u];   // depth value
                if (d == 0)     // this point is not measured
                    continue;
                Eigen::Vector3d point_img;  // image coordinate system
                point_img[2] = double(d) / depth_scale;
                point_img[0] = (u - cx) * point_img[2] / fx;
                point_img[1] = (v - cy) * point_img[2] / fy;
                Eigen::Vector3d point_world = T * point_img;    // world coordinate system
                PointT point_data;
                point_data.x = point_world[0];
                point_data.y = point_world[1];
                point_data.z = point_world[2];
                point_data.b = color.data[v * color.step + u * color.channels()];
                point_data.g = color.data[v * color.step + u * color.channels() + 1];
                point_data.b = color.data[v * color.step + u * color.channels() + 2];
                point_cloud -> points.push_back(point_data);
            }
    }
    point_cloud -> is_dense = false;
    std::cout << "Point cloud count: " << point_cloud->size() << std::endl;
    pcl::io::savePCDFileBinary(CLOUD_SAVE_PATH, *point_cloud);

    // to view the point cloud
    // pcl_viewer map.pcd
}

void imageBasics()
{
    // read image
    cv::Mat image;
    image = cv::imread(IMAGE_PATH);
    if (image.data == nullptr){
        std::cerr << "File not exist! " << std::endl;
        return;
    }

    // input basic info
    std::cout << "width: " << image.cols << ", height: "
        << image.rows << ", channels: " << image.channels() << std::endl;
    cv::imshow("image", image);
    cv::waitKey(0);     // pause, wait for a key
    // check image type
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3){
        // image is not qualified
        std::cout << "Image type error!" << std::endl;
        return;
    }

    // access image element
    // use std::chrono to time
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < image.rows; ++y){
        for (size_t x = 0; x < image.cols; ++x){
            // use cv::Mat::ptr to get row pointer
            unsigned char* row_ptr = image.ptr<unsigned char>(y);//image.ptr<unsigned char>(y);
            unsigned char* data_ptr = &row_ptr[x * image.channels()];
            // print every channel
            for (int c = 0; c != image.channels(); ++c){
                // data is the c-th channel value of one pixel
                unsigned char data = data_ptr[c];
            }
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Access image element cost " << time_used.count() << " seconds." << std::endl;

    // After assignment, it is the same image object
    cv::Mat image_another = image;
    // Modifications will propagate
    image_another(cv::Rect(0,0,100,100)).setTo(0);
    cv::imshow("image", image);
    cv::waitKey(0);

    // After clone, it is NOT the same image object
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
//    imageBasics();
    joinMap();
    return 0;
}

