cmake_minimum_required(VERSION 2.8)
project(chapter01)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)

CUDA_ADD_EXECUTABLE(heat heat_emulator.cu)

target_link_libraries(heat ${OpenCV_LIBS})
