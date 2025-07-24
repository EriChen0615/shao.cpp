#include <iostream>
#include <vector>
#include <cmath>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    std::cout << "Testing Automatic Differentiation with Non-trivial Computational Graph..." << std::endl;
    
    // Create a computational graph: (a + b) + (c + d) + e
    // This creates a graph with multiple paths and intermediate nodes
    
    // Input tensors
    shao::Tensor<float> a({1.0f, 2.0f, 3.0f});
    shao::Tensor<float> b({4.0f, 5.0f, 6.0f});
    shao::Tensor<float> c({7.0f, 8.0f, 9.0f});
    shao::Tensor<float> d({10.0f, 11.0f, 12.0f});
    shao::Tensor<float> e({13.0f, 14.0f, 15.0f});
    
    std::cout << "Input tensors:" << std::endl;
    a.print_info();
    b.print_info();
    c.print_info();
    d.print_info();
    e.print_info();
    
    // Build computational graph: (a + b) + (c + d) + e
    shao::AddTensorOp<float> add_op;
    
    // First level: a + b and c + d
    shao::Tensor<float> ab = add_op(a, b);
    shao::Tensor<float> cd = add_op(c, d);
    
    std::cout << "\nIntermediate tensors (first level):" << std::endl;
    ab.print_info();
    cd.print_info();
    
    // Second level: (a + b) + (c + d)
    shao::Tensor<float> ab_cd = add_op(ab, cd);
    
    std::cout << "\nIntermediate tensor (second level):" << std::endl;
    ab_cd.print_info();
    
    // Final result: (a + b) + (c + d) + e
    shao::Tensor<float> result = add_op(ab_cd, e);
    
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
    // Expected: (1+4) + (7+10) + 13 = 5 + 17 + 13 = 35
    //           (2+5) + (8+11) + 14 = 7 + 19 + 14 = 40  
    //           (3+6) + (9+12) + 15 = 9 + 21 + 15 = 45
    std::vector<float> expected = {35.0f, 40.0f, 45.0f};
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
    std::cout << "\n=== Testing Automatic Differentiation ===" << std::endl;
    
    // Initialize gradient for the output (ones tensor)
    std::vector<float> ones(result_data.size(), 1.0f);
    result.set_grad(std::make_shared<shao::Tensor<float>>(ones, result.device()));
    
    std::cout << "Initialized output gradient (ones tensor)" << std::endl;
    
    // Run backward pass
    std::cout << "\nRunning backward pass..." << std::endl;
    result.backward();
    
    // Check gradients for input tensors
    std::cout << "\nChecking gradients for input tensors:" << std::endl;
    
    // For this graph: ∂result/∂a = ∂result/∂b = ∂result/∂c = ∂result/∂d = ∂result/∂e = 1
    // Each input contributes directly to the final result through addition
    
    std::vector<shao::Tensor<float>*> inputs = {&a, &b, &c, &d, &e};
    std::vector<std::string> input_names = {"a", "b", "c", "d", "e"};
    
    bool gradients_correct = true;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* input = inputs[i];
        const std::string& name = input_names[i];
        
        if (input->grad()) {
            input->grad()->realize();
            const auto& grad_data = input->grad()->data();
            
            std::cout << "Gradient for " << name << ": ";
            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << grad_data[j];
            }
            std::cout << std::endl;
            
            // Check that gradient is all ones (∂result/∂input = 1 for addition)
            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (std::abs(grad_data[j] - 1.0f) > 1e-6) {
                    std::cout << "✗ Gradient error for " << name << " at index " << j 
                              << ": expected 1.0, got " << grad_data[j] << std::endl;
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
    
    std::vector<shao::Tensor<float>*> intermediates = {&ab, &cd, &ab_cd};
    std::vector<std::string> intermediate_names = {"ab", "cd", "ab_cd"};
    
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
    std::cout << "✓ Forward computation: (a + b) + (c + d) + e" << std::endl;
    std::cout << "✓ Backward computation: ∂result/∂input = 1 for all inputs" << std::endl;
    std::cout << "✓ Computational graph with multiple paths and intermediate nodes" << std::endl;
    std::cout << "✓ Automatic differentiation working correctly!" << std::endl;
    
    return 0;
} 