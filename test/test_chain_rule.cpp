#include <iostream>
#include <vector>
#include <cmath>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    std::cout << "Testing Chain Rule with Addition and Multiplication..." << std::endl;
    
    // Create a computational graph: (a * b) + (c * d)
    // This will show different gradients for multiplication vs addition
    
    // Input tensors
    shao::Tensor<float> a({2.0f, 3.0f, 4.0f});
    shao::Tensor<float> b({5.0f, 6.0f, 7.0f});
    shao::Tensor<float> c({8.0f, 9.0f, 10.0f});
    shao::Tensor<float> d({11.0f, 12.0f, 13.0f});
    
    std::cout << "Input tensors:" << std::endl;
    a.print_info();
    b.print_info();
    c.print_info();
    d.print_info();
    
    // Build computational graph: (a * b) + (c * d)
    shao::AddTensorOp<float> add_op;
    shao::MulTensorOp<float> mul_op;
    
    // First level: a * b and c * d
    shao::Tensor<float> ab = mul_op(a, b);
    shao::Tensor<float> cd = mul_op(c, d);
    
    std::cout << "\nIntermediate tensors (first level):" << std::endl;
    ab.print_info();
    cd.print_info();
    
    // Final result: (a * b) + (c * d)
    shao::Tensor<float> result = add_op(ab, cd);
    
    std::cout << "\nFinal result tensor:" << std::endl;
    result.print_info();
    
    // Realize the computation
    result.realize();
    
    std::cout << "\nComputed result: ";
    const auto& result_data = result.data();
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result_data[i];
    }
    std::cout << std::endl;
    
    // Verify the computation manually
    // Expected: (2*5) + (8*11) = 10 + 88 = 98
    //           (3*6) + (9*12) = 18 + 108 = 126  
    //           (4*7) + (10*13) = 28 + 130 = 158
    std::vector<float> expected = {98.0f, 126.0f, 158.0f};
    bool computation_correct = true;
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (std::abs(result_data[i] - expected[i]) > 1e-6) {
            std::cout << "✗ Computation error at index " << i << ": expected " << expected[i] 
                      << ", got " << result_data[i] << std::endl;
            computation_correct = false;
        }
    }
    
    if (computation_correct) {
        std::cout << "✓ Forward computation is correct!" << std::endl;
    } else {
        std::cout << "✗ Forward computation failed!" << std::endl;
        return 1;
    }
    
    // Test automatic differentiation
    std::cout << "\n=== Testing Chain Rule ===" << std::endl;
    
    // Initialize gradient for the output (ones tensor)
    std::vector<float> ones(result_data.size(), 1.0f);
    result.set_grad(std::make_shared<shao::Tensor<float>>(ones, result.device()));
    
    std::cout << "Initialized output gradient (ones tensor)" << std::endl;
    
    // Run backward pass
    std::cout << "\nRunning backward pass..." << std::endl;
    result.backward();
    
    // Check gradients for input tensors
    std::cout << "\nChecking gradients for input tensors:" << std::endl;
    
    // For this graph: 
    // ∂result/∂a = ∂result/∂(a*b) * ∂(a*b)/∂a = 1 * b = b
    // ∂result/∂b = ∂result/∂(a*b) * ∂(a*b)/∂b = 1 * a = a
    // ∂result/∂c = ∂result/∂(c*d) * ∂(c*d)/∂c = 1 * d = d
    // ∂result/∂d = ∂result/∂(c*d) * ∂(c*d)/∂d = 1 * c = c
    
    std::vector<shao::Tensor<float>*> inputs = {&a, &b, &c, &d};
    std::vector<std::string> input_names = {"a", "b", "c", "d"};
    std::vector<std::vector<float>> expected_gradients = {
        {5.0f, 6.0f, 7.0f},  // ∂result/∂a = b
        {2.0f, 3.0f, 4.0f},  // ∂result/∂b = a
        {11.0f, 12.0f, 13.0f}, // ∂result/∂c = d
        {8.0f, 9.0f, 10.0f}   // ∂result/∂d = c
    };
    
    bool gradients_correct = true;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* input = inputs[i];
        const std::string& name = input_names[i];
        const std::vector<float>& expected_grad = expected_gradients[i];
        
        if (input->grad()) {
            input->grad()->realize();
            const auto& grad_data = input->grad()->data();
            
            std::cout << "Gradient for " << name << ": ";
            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << grad_data[j];
            }
            std::cout << std::endl;
            
            // Check that gradient matches expected
            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (std::abs(grad_data[j] - expected_grad[j]) > 1e-6) {
                    std::cout << "✗ Gradient error for " << name << " at index " << j 
                              << ": expected " << expected_grad[j] << ", got " << grad_data[j] << std::endl;
                    gradients_correct = false;
                }
            }
        } else {
            std::cout << "✗ No gradient computed for " << name << std::endl;
            gradients_correct = false;
        }
    }
    
    if (gradients_correct) {
        std::cout << "\n✓ All gradients are correct!" << std::endl;
    } else {
        std::cout << "\n✗ Gradient computation failed!" << std::endl;
        return 1;
    }
    
    // Test intermediate tensor gradients
    std::cout << "\nChecking gradients for intermediate tensors:" << std::endl;
    
    std::vector<shao::Tensor<float>*> intermediates = {&ab, &cd};
    std::vector<std::string> intermediate_names = {"ab", "cd"};
    
    for (size_t i = 0; i < intermediates.size(); ++i) {
        auto* intermediate = intermediates[i];
        const std::string& name = intermediate_names[i];
        
        if (intermediate->grad()) {
            intermediate->grad()->realize();
            const auto& grad_data = intermediate->grad()->data();
            
            std::cout << "Gradient for " << name << ": ";
            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << grad_data[j];
            }
            std::cout << std::endl;
        } else {
            std::cout << "No gradient for " << name << " (this is expected for intermediate nodes)" << std::endl;
        }
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✓ Forward computation: (a * b) + (c * d)" << std::endl;
    std::cout << "✓ Chain rule: ∂result/∂a = b, ∂result/∂b = a, ∂result/∂c = d, ∂result/∂d = c" << std::endl;
    std::cout << "✓ Different gradients for multiplication vs addition operations" << std::endl;
    std::cout << "✓ Chain rule multiplication working correctly!" << std::endl;
    
    return 0;
} 