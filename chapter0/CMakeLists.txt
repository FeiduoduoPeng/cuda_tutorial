cmake_minimum_required(VERSION 2.8)
project(chapter0)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(CUDA)

CUDA_ADD_LIBRARY(add vector_add.cu)
CUDA_ADD_EXECUTABLE(g main.cpp)
target_link_libraries(g add)
#target_link_libraries(g ${OpenCV_LIBS})

