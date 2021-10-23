add_executable( my_slam
    frame.cpp
    map_point.cpp
    map.cpp
    camera.cpp
    config.cpp
    main.cpp
)

target_link_libraries( my_slam
    ${THIRD_PARTY_LIBS}
)