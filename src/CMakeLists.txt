# Create the shao library as shared library
cuda_add_library(shao_lib SHARED
    shao/tensor.cpp
    shao/op.cpp
    shao/op.cu
)

# Set include directories
target_include_directories(shao_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/headers
    ${CUDA_INCLUDE_DIRS}
)

# Link CUDA libraries to the library itself
target_link_libraries(shao_lib ${CUDA_LIBRARIES}) 