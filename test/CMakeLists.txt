# Create test executables
add_executable(test_tensor_cpu test_tensor_cpu.cpp)
add_executable(test_tensor_gpu test_tensor_gpu.cpp)

# Link with shao library and CUDA libraries
target_link_libraries(test_tensor_cpu shao_lib)
target_link_libraries(test_tensor_gpu shao_lib ${CUDA_LIBRARIES}) 