#include "my_slam/common_include.h"
#include "my_slam/config.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    my_slam::Config::setParameterFile(CONFIG_PATH);
    auto fx = my_slam::Config::get<double>("camera.fx");
    std::cout << "fx: " << fx << std::endl;
    return 0;
}
