//
// Created by zavier on 11/2/21.
//
#include "loop_closure.h"


//#define FILE_NAME "vocabulary.yml.gz"
#define FILE_NAME "vocab_larger.yml.gz"
// TODO: why use larger vocabulary the result didn't get better???

void calcSimilarity()
{
    // read the images and database
    std::cout << "Reading database" << std::endl;
    DBoW3::Vocabulary vocab(FILE_NAME);
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want:
    if (vocab.empty()) {
        std::cerr << "Vocabulary does not exist." << std::endl;
        return;
    }
    std::cout << "Reading images... " << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; i++) {
        std::string path = "./data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // NOTE: in this case we are comparing images with a vocabulary generated by themselves, this may lead to overfit.
    // detect ORB features
    std::cout << "Detecting ORB features ... " << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat & image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // we can compare the images directly, or we can compare one image to a database
    // images :
    std::cout << "Comparing images with images " << std::endl;
    for (int i = 0; i < images.size(); ++i) {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); j++) {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            std::cout << "image " << i << " vs image " << j << " : " << score << std::endl;
        }
        std::cout << std::endl;
    }

    // or compare with database
    std::cout << "Comparing images with database " << std::endl;
    DBoW3::Database db(vocab, false, 0);
    for (auto & descriptor : descriptors)
        db.add(descriptor);
    std::cout << "Database info: " << db << std::endl;
    for (int i = 0; i < descriptors.size(); ++i) {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);      // max result=4
        std::cout << "Searching for image " << i << " returns " << ret << std::endl << std::endl;
    }
    std::cout << "Done." << std::endl;
}

