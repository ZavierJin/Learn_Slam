#include <iostream>
#include <chrono> // timing
#include <fstream>
#include <boost/format.hpp>

#include <opencv2/core/core.hpp> // version 2 ??
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#define IMAGE_PATH "ubuntu.png"
//#define IMAGE_PATH "/home/zavier/cpp_code/test1/ubuntu.png"

void joinMap()
{

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
    imageBasics();
    return 0;
}

