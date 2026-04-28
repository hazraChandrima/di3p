cmake_minimum_required(VERSION 3.16)
project(image_enhancer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(enhancer
    main.cpp
    src/dft.cpp
    src/kmeans.cpp
    src/correction.cpp
    src/metrics.cpp
)

target_compile_options(enhancer PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
)
