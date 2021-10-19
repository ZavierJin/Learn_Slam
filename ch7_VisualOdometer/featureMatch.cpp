//
// Created by zavier on 10/17/21.
//
#include "visual_odometer.h"


void featureMatch(const cv::Mat& img_1, const cv::Mat& img_2,
                  std::vector<cv::KeyPoint>& keypoint_1,
                  std::vector<cv::KeyPoint>& keypoint_2,
                  std::vector<cv::DMatch>& matches)
{
    cv::Mat descriptor_1, descriptor_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //-- Detect Oriented FAST corner position
    detector->detect(img_1,keypoint_1);
    detector->detect(img_2,keypoint_2);

    //-- Compute BRIEF descriptor based on keypoint
    descriptor->compute(img_1, keypoint_1, descriptor_1);
    descriptor->compute(img_2, keypoint_2, descriptor_2);

    //-- Match the BRIEF descriptors in the two images, using Hamming distance
    std::vector<cv::DMatch> tmp_match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptor_1, descriptor_2, tmp_match);

    //-- Filter the matching point pairs
    double min_dist = 10000, max_dist = 0;
    // Find out the minimum distance and maximum distance between all matches
    // The distance between the most similar and the least similar two groups of points
    for (int i = 0; i < descriptor_1.rows; ++i) {
        double dist = tmp_match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    std::cout << "Max dist: " << max_dist << std::endl;
    std::cout << "Min dist: " << min_dist << std::endl;
    // When the distance between descriptors is greater than twice the minimum distance,
    // it is considered that the matching is wrong.
    // But sometimes the minimum distance is very small.
    // Set an empirical value as the lower limit.
    double threshold = 30.0;
    for (int i = 0; i < descriptor_1.rows; ++i) {
        if (tmp_match[i].distance <= cv::max(2*min_dist, threshold))
            matches.push_back(tmp_match[i]);
    }
}

__attribute__((unused)) void featureExtraction()
{
    cv::Mat img_1 = cv::imread(IMG_PATH_1, CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(IMG_PATH_2, CV_LOAD_IMAGE_COLOR);

    //-- Initialization
    std::vector<cv::KeyPoint> keypoint_1, keypoint_2;
    cv::Mat descriptor_1, descriptor_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0,
                                           2, cv::ORB::HARRIS_SCORE, 31, 20);

    //-- Detect Oriented FAST corner position
    orb->detect(img_1, keypoint_1);
    orb->detect(img_2, keypoint_2);

    //-- Compute BRIEF descriptor based on keypoint
    orb->compute(img_1, keypoint_1, descriptor_1);
    orb->compute(img_2, keypoint_2, descriptor_2);
    cv::Mat out_img_1;
    cv::drawKeypoints(img_1, keypoint_1, out_img_1,
                      cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB Keypoint", out_img_1);

    //-- Match the BRIEF descriptors in the two images, using Hamming distance
    std::vector<cv::DMatch> matches;    // store result
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor_1, descriptor_2, matches);

    //-- Filter the matching point pairs
    double min_dist = 10000, max_dist = 0;
    // Find out the minimum distance and maximum distance between all matches
    // The distance between the most similar and the least similar two groups of points
    for (int i = 0; i < descriptor_1.rows; ++i) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    std::cout << "Max dist: " << max_dist << std::endl;
    std::cout << "Min dist: " << min_dist << std::endl;
    // When the distance between descriptors is greater than twice the minimum distance,
    // it is considered that the matching is wrong.
    // But sometimes the minimum distance is very small.
    // Set an empirical value as the lower limit.
    std::vector<cv::DMatch> good_matches;
    double threshold = 30.0;
    for (int i = 0; i < descriptor_1.rows; ++i) {
        if (matches[i].distance <= cv::max(2*min_dist, threshold))
            good_matches.push_back(matches[i]);
    }

    //-- Draw matching result
    cv::Mat img_match;
    cv::Mat img_good_match;
    cv::drawMatches(img_1, keypoint_1, img_2, keypoint_2, matches, img_match);
    cv::drawMatches(img_1, keypoint_1, img_2, keypoint_2, good_matches, img_good_match);
    cv::imshow("Matches", img_match);
    cv::imshow("Good Matches", img_good_match);
    cv::waitKey(0);

}
