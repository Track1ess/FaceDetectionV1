cmake_minimum_required(VERSION 3.10)
project(faceDetection)

set(CMAKE_CXX_STANDARD 14)

# System OpenCV only
include_directories(/usr/include/opencv4)
link_directories(/usr/lib)  # or /usr/lib/aarch64-linux-gnu on Jetson

add_executable(faceDetection main.cpp)

target_link_libraries(faceDetection
        PRIVATE
        opencv_core
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        opencv_videoio
)
