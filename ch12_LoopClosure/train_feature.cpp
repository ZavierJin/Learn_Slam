//
// Created by zavier on 11/2/21.
//
/*
 * Train dictionary using DBoW3
 */

#include "loop_closure.h"

void trainFeature()
{
    // read the image
    std::cout << "reading images... " << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i) {
        std::string path = "./data/" + std::to_string(i+1) + ".png";
        images.push_back(cv::imread(path));
    }
    // detect ORB features
    std::cout << "detecting ORB features ... " << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat& image:images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute( image, cv::Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
    // create vocabulary
    std::cout << "creating vocabulary ... " << std::endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    std::cout << "vocabulary info: " << vocab << std::endl;
    vocab.save( "vocabulary.yml.gz" );
    std::cout << "done" << std::endl;
}

