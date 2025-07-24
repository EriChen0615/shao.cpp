cmake_minimum_required(VERSION 3.16)
project(shao)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA properties for A5000 (Ampere, compute capability 8.6)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_86)

# Add source directory
add_subdirectory(src)

# Add test directory
add_subdirectory(test)