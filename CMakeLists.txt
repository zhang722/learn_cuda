cmake_minimum_required(VERSION 3.10)

project(learn_cuda LANGUAGES CUDA C CXX)

add_executable(
    ${PROJECT_NAME}
    src/test.cu
)

add_executable(
    template

    src/template.cu
)

add_executable(
    scan

    src/scan/scan.cu
)

add_executable(
    cpu

    src/main.cpp
)