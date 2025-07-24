#include <iostream>
#include <vector>
#include <chrono>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    // Create large tensors (1 million elements)
    const int size = 1000000;
    std::vector<float> data_a(size);
    std::vector<float> data_b(size);
    
    // Initialize with some values
    for (int i = 0; i < size; ++i) {
        data_a[i] = static_cast<float>(i);
        data_b[i] = static_cast<float>(i * 2);
    }
    
    shao::Tensor<float> a(data_a, shao::Device::CUDA);
    shao::Tensor<float> b(data_b, shao::Device::CUDA);

    shao::AddTensorOp<float> add_op;
    
    // Warm up
    auto warmup = add_op(a, b);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Repeat computation 50 times with different data
    float sum = 0.0f;
    for (int i = 0; i < 50; ++i) {
        // Modify data slightly each iteration to prevent optimization
        for (int j = 0; j < size; ++j) {
            data_a[j] = static_cast<float>(i + j);
            data_b[j] = static_cast<float>((i + j) * 2);
        }
        shao::Tensor<float> a_iter(data_a, shao::Device::CUDA);
        shao::Tensor<float> b_iter(data_b, shao::Device::CUDA);
        
        auto c = add_op(a_iter, b_iter);
        
        // Use the result to prevent optimization
        sum += c.data()[0] + c.data()[size-1];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CUDA: 50 iterations with " << size << "-dimensional tensors" << std::endl;
    std::cout << "Total time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Average time per iteration: " << duration.count() / 50.0 << " microseconds" << std::endl;
    
    // Verify result
    auto final_result = add_op(a, b);
    std::cout << "Sum of first and last elements across all iterations: " << sum << std::endl;
    std::cout << "First 5 elements of final result: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << final_result.data()[i] << " ";
    }
    std::cout << std::endl;

    return 0;
} 