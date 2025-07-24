# Create test executable
add_executable(test_tensor test_tensor.cpp)

# Link with shao library
target_link_libraries(test_tensor shao_lib) 