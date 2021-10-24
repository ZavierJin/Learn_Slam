//
// Created by zavier on 10/23/21.
//
#include "my_slam/config.h"

namespace my_slam
{
void Config::setParameterFile( const std::string& filename)
{
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage( filename.c_str(), cv::FileStorage::READ );
    if (!config_->file_.isOpened()) {
        std::cerr<<"parameter file "<<filename<<" does not exist."<<std::endl;
        config_->file_.release();
        return;
    }
}

Config::~Config()
{
    if (file_.isOpened())
        file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

}
