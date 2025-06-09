cmake_minimum_required(VERSION 3.10)

project(learn_cuda LANGUAGES CUDA C CXX)

set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/scan
)

add_executable(
    ${PROJECT_NAME}
    src/test.cu
)



add_executable(
    cudaScan

    src/scan/scan.cu
    src/scan/options.cpp
)

add_executable(
    cpu

    src/main.cpp
)