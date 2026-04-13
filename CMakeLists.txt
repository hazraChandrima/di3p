cmake_minimum_required(VERSION 3.14)
project(ImageQualityAnalyzer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: enable optimisations
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenCV (used only for image I/O and display)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# All headers are in src/
include_directories(${CMAKE_SOURCE_DIR})

add_executable(analyzer src/main.cpp)
target_link_libraries(analyzer ${OpenCV_LIBS})
