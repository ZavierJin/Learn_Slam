# Declare the minimum version of cmake required
cmake_minimum_required(VERSION 2.8)

# Declare a cmake project
project(HelloSLAM)

# # add library, [.a]
# add_library(hello libHelloSLAM.cpp)

# add shared library, [.so]
add_library(hello_shared SHARED libHelloSLAM.cpp)

# add an executable program
add_executable(helloSLAM helloSLAM.cpp)
add_executable(useHello useHello.cpp)

# link library, should write after add_executable
target_link_libraries(useHello hello_shared)

