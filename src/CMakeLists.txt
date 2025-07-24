# Create the shao library
add_library(shao_lib
    shao/tensor.cpp
    shao/op.cpp
)

# Set include directories
target_include_directories(shao_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 