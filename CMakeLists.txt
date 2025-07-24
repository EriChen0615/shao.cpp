cmake_minimum_required(VERSION 3.10)
project(shao)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source directory
add_subdirectory(src)

# Add test directory
add_subdirectory(test)