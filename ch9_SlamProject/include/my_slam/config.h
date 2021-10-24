//
// Created by zavier on 10/23/21.
//

#ifndef MY_SLAM_CONFIG_H
#define MY_SLAM_CONFIG_H

#include "my_slam/common_include.h"

namespace my_slam
{
class Config
{
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;

    Config () = default; // private constructor makes a singleton
public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static void setParameterFile( const std::string& filename );

    // access the parameter values
    template< typename T >
    static T get( const std::string& key )
    {
        return T(Config::config_->file_[key]);
    }
};
}

#endif //MY_SLAM_CONFIG_H
