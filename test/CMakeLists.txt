# Create test executables
add_executable(test_tensor_cpu test_tensor_cpu.cpp)
add_executable(test_tensor_gpu test_tensor_gpu.cpp)
add_executable(test_sum_op test_sum_op.cpp)
add_executable(test_mul_op test_mul_op.cpp)
add_executable(test_autodiff test_autodiff.cpp)
add_executable(test_chain_rule test_chain_rule.cpp)

# Link with shao library and CUDA libraries
target_link_libraries(test_tensor_cpu shao_lib)
target_link_libraries(test_tensor_gpu shao_lib ${CUDA_LIBRARIES})
target_link_libraries(test_sum_op shao_lib)
target_link_libraries(test_mul_op shao_lib)
target_link_libraries(test_autodiff shao_lib)
target_link_libraries(test_chain_rule shao_lib) 