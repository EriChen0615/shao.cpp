#include <iostream>
#include <vector>
#include <cmath>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    std::cout << "Testing SumTensorOp functionality..." << std::endl;
    
    // Create test tensors
    shao::Tensor<float> t1({1.0f, 2.0f, 3.0f, 4.0f});
    shao::Tensor<float> t2({5.0f, 6.0f, 7.0f, 8.0f});
    shao::Tensor<float> t3({9.0f, 10.0f, 11.0f, 12.0f});
    
    std::cout << "Input tensors:" << std::endl;
    t1.print_info();
    t2.print_info();
    t3.print_info();
    
    // Test CPU sum operation
    shao::SumTensorOp<float> sum_op;
    std::vector<shao::Tensor<float>*> tensors = {&t1, &t2, &t3};
    shao::Tensor<float> result = sum_op(tensors);
    
    std::cout << "\nSum result tensor:" << std::endl;
    result.print_info();
    
    // Realize the result
    result.realize();
    
    std::cout << "\nSum result data: ";
    const auto& result_data = result.data();
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result_data[i];
    }
    std::cout << std::endl;
    
    // Verify the result
    std::vector<float> expected = {15.0f, 18.0f, 21.0f, 24.0f};  // 1+5+9, 2+6+10, 3+7+11, 4+8+12
    bool correct = true;
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (std::abs(result_data[i] - expected[i]) > 1e-6) {
            std::cout << "✗ Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << result_data[i] << std::endl;
            correct = false;
        }
    }
    
    if (correct) {
        std::cout << "✓ CPU sum operation works correctly!" << std::endl;
    } else {
        std::cout << "✗ CPU sum operation failed!" << std::endl;
        return 1;
    }
    
    // Test with different number of tensors
    std::cout << "\nTesting sum with 2 tensors..." << std::endl;
    std::vector<shao::Tensor<float>*> two_tensors = {&t1, &t2};
    shao::Tensor<float> result2 = sum_op(two_tensors);
    result2.realize();
    
    std::cout << "Sum of 2 tensors: ";
    const auto& result2_data = result2.data();
    for (size_t i = 0; i < result2_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result2_data[i];
    }
    std::cout << std::endl;
    
    // Test with single tensor
    std::cout << "\nTesting sum with 1 tensor..." << std::endl;
    std::vector<shao::Tensor<float>*> one_tensor = {&t1};
    shao::Tensor<float> result3 = sum_op(one_tensor);
    result3.realize();
    
    std::cout << "Sum of 1 tensor: ";
    const auto& result3_data = result3.data();
    for (size_t i = 0; i < result3_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result3_data[i];
    }
    std::cout << std::endl;
    
    std::cout << "\nAll sum operation tests passed!" << std::endl;
    return 0;
} 