cmake_minimum_required(VERSION 2.8)
project(chapter01)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
find_package(OpenCV REQUIRED) 
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)

CUDA_ADD_EXECUTABLE(add vector_add.cu)
CUDA_ADD_EXECUTABLE(g julia_gpu.cu)
target_link_libraries(g ${OpenCV_LIBS})

