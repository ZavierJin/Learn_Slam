add_executable( my_slam
    ../test/main.cpp
    frame.cpp
    map_point.cpp
    map.cpp
    camera.cpp
    config.cpp
    visual_odometer_2.cpp
    g2o_type.cpp
)

target_link_libraries( my_slam
    ${THIRD_PARTY_LIBS}
)