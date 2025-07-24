#include <iostream>
#include <vector>
#include <cmath>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    std::cout << "Testing MulTensorOp functionality..." << std::endl;
    
    // Create test tensors
    shao::Tensor<float> t1({1.0f, 2.0f, 3.0f, 4.0f});
    shao::Tensor<float> t2({5.0f, 6.0f, 7.0f, 8.0f});
    
    std::cout << "Input tensors:" << std::endl;
    t1.print_info();
    t2.print_info();
    
    // Test CPU multiplication operation
    shao::MulTensorOp<float> mul_op;
    shao::Tensor<float> result = mul_op(t1, t2);
    
    std::cout << "\nMultiplication result tensor:" << std::endl;
    result.print_info();
    
    // Realize the result
    result.realize();
    
    std::cout << "\nMultiplication result data: ";
    const auto& result_data = result.data();
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result_data[i];
    }
    std::cout << std::endl;
    
    // Verify the result
    std::vector<float> expected = {5.0f, 12.0f, 21.0f, 32.0f};  // 1*5, 2*6, 3*7, 4*8
    bool correct = true;
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (std::abs(result_data[i] - expected[i]) > 1e-6) {
            std::cout << "✗ Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << result_data[i] << std::endl;
            correct = false;
        }
    }
    
    if (correct) {
        std::cout << "✓ CPU multiplication operation works correctly!" << std::endl;
    } else {
        std::cout << "✗ CPU multiplication operation failed!" << std::endl;
        return 1;
    }
    
    // Test with different values
    std::cout << "\nTesting multiplication with different values..." << std::endl;
    shao::Tensor<float> t3({0.5f, 1.5f, 2.5f, 3.5f});
    shao::Tensor<float> t4({2.0f, 4.0f, 6.0f, 8.0f});
    
    shao::Tensor<float> result2 = mul_op(t3, t4);
    result2.realize();
    
    std::cout << "Multiplication of different values: ";
    const auto& result2_data = result2.data();
    for (size_t i = 0; i < result2_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result2_data[i];
    }
    std::cout << std::endl;
    
    // Verify the second result
    std::vector<float> expected2 = {1.0f, 6.0f, 15.0f, 28.0f};  // 0.5*2, 1.5*4, 2.5*6, 3.5*8
    bool correct2 = true;
    
    for (size_t i = 0; i < result2_data.size(); ++i) {
        if (std::abs(result2_data[i] - expected2[i]) > 1e-6) {
            std::cout << "✗ Mismatch at index " << i << ": expected " << expected2[i] 
                      << ", got " << result2_data[i] << std::endl;
            correct2 = false;
        }
    }
    
    if (correct2) {
        std::cout << "✓ Second multiplication test passed!" << std::endl;
    } else {
        std::cout << "✗ Second multiplication test failed!" << std::endl;
        return 1;
    }
    
    // Test with zero values
    std::cout << "\nTesting multiplication with zero..." << std::endl;
    shao::Tensor<float> t5({1.0f, 2.0f, 3.0f, 4.0f});
    shao::Tensor<float> t6({0.0f, 0.0f, 0.0f, 0.0f});
    
    shao::Tensor<float> result3 = mul_op(t5, t6);
    result3.realize();
    
    std::cout << "Multiplication with zero: ";
    const auto& result3_data = result3.data();
    for (size_t i = 0; i < result3_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << result3_data[i];
    }
    std::cout << std::endl;
    
    // Verify zero result
    std::vector<float> expected3 = {0.0f, 0.0f, 0.0f, 0.0f};
    bool correct3 = true;
    
    for (size_t i = 0; i < result3_data.size(); ++i) {
        if (std::abs(result3_data[i] - expected3[i]) > 1e-6) {
            std::cout << "✗ Mismatch at index " << i << ": expected " << expected3[i] 
                      << ", got " << result3_data[i] << std::endl;
            correct3 = false;
        }
    }
    
    if (correct3) {
        std::cout << "✓ Zero multiplication test passed!" << std::endl;
    } else {
        std::cout << "✗ Zero multiplication test failed!" << std::endl;
        return 1;
    }
    
    // Test partial adjoint (automatic differentiation)
    std::cout << "\nTesting partial adjoint for multiplication..." << std::endl;
    
    // Create tensors for differentiation test
    shao::Tensor<float> diff_a({2.0f, 3.0f, 4.0f});
    shao::Tensor<float> diff_b({5.0f, 6.0f, 7.0f});
    
    // Create multiplication operation
    shao::Tensor<float> product = mul_op(diff_a, diff_b);
    product.realize();
    
    // Test partial derivative with respect to 'diff_a'
    std::vector<shao::Tensor<float>*> inputs = {&diff_a, &diff_b};
    auto partial_a = mul_op.partial_adjoint(inputs, &product, diff_a.id());
    
    std::cout << "Partial derivative ∂(a*b)/∂a: ";
    const auto& partial_a_data = partial_a->data();
    for (size_t i = 0; i < partial_a_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << partial_a_data[i];
    }
    std::cout << std::endl;
    
    // Verify partial derivative with respect to 'diff_a' should be 'diff_b'
    std::vector<float> expected_partial_a = {5.0f, 6.0f, 7.0f};
    bool correct_partial_a = true;
    
    for (size_t i = 0; i < partial_a_data.size(); ++i) {
        if (std::abs(partial_a_data[i] - expected_partial_a[i]) > 1e-6) {
            std::cout << "✗ Partial derivative mismatch at index " << i << ": expected " << expected_partial_a[i] 
                      << ", got " << partial_a_data[i] << std::endl;
            correct_partial_a = false;
        }
    }
    
    if (correct_partial_a) {
        std::cout << "✓ Partial derivative ∂(a*b)/∂a = b is correct!" << std::endl;
    } else {
        std::cout << "✗ Partial derivative test failed!" << std::endl;
        return 1;
    }
    
    // Test partial derivative with respect to 'diff_b'
    auto partial_b = mul_op.partial_adjoint(inputs, &product, diff_b.id());
    
    std::cout << "Partial derivative ∂(a*b)/∂b: ";
    const auto& partial_b_data = partial_b->data();
    for (size_t i = 0; i < partial_b_data.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << partial_b_data[i];
    }
    std::cout << std::endl;
    
    // Verify partial derivative with respect to 'diff_b' should be 'diff_a'
    std::vector<float> expected_partial_b = {2.0f, 3.0f, 4.0f};
    bool correct_partial_b = true;
    
    for (size_t i = 0; i < partial_b_data.size(); ++i) {
        if (std::abs(partial_b_data[i] - expected_partial_b[i]) > 1e-6) {
            std::cout << "✗ Partial derivative mismatch at index " << i << ": expected " << expected_partial_b[i] 
                      << ", got " << partial_b_data[i] << std::endl;
            correct_partial_b = false;
        }
    }
    
    if (correct_partial_b) {
        std::cout << "✓ Partial derivative ∂(a*b)/∂b = a is correct!" << std::endl;
    } else {
        std::cout << "✗ Partial derivative test failed!" << std::endl;
        return 1;
    }
    
    // Test gradient computation with automatic differentiation
    std::cout << "\n=== Testing Gradient Computation ===" << std::endl;
    
    // Create tensors for gradient test
    shao::Tensor<float> x({2.0f, 3.0f, 4.0f});
    shao::Tensor<float> y({5.0f, 6.0f, 7.0f});
    
    // Create multiplication: z = x * y
    shao::Tensor<float> z = mul_op(x, y);
    z.realize();
    
    std::cout << "x = [2, 3, 4]" << std::endl;
    std::cout << "y = [5, 6, 7]" << std::endl;
    std::cout << "z = x * y = [10, 18, 28]" << std::endl;
    
    // Initialize gradient for output (ones tensor)
    std::vector<float> ones(z.data().size(), 1.0f);
    z.set_grad(std::make_shared<shao::Tensor<float>>(ones, z.device()));
    
    std::cout << "Initialized output gradient (ones tensor)" << std::endl;
    
    // Run backward pass
    std::cout << "\nRunning backward pass..." << std::endl;
    z.backward();
    
    // Check gradients for input tensors
    std::cout << "\nChecking gradients for input tensors:" << std::endl;
    
    // For z = x * y:
    // ∂z/∂x = y = [5, 6, 7]
    // ∂z/∂y = x = [2, 3, 4]
    
    bool gradients_correct = true;
    
    // Check gradient for x
    if (x.grad()) {
        x.grad()->realize();
        const auto& x_grad_data = x.grad()->data();
        
        std::cout << "Gradient for x (∂z/∂x): ";
        for (size_t i = 0; i < x_grad_data.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << x_grad_data[i];
        }
        std::cout << std::endl;
        
        // Expected: ∂z/∂x = y = [5, 6, 7]
        std::vector<float> expected_x_grad = {5.0f, 6.0f, 7.0f};
        for (size_t i = 0; i < x_grad_data.size(); ++i) {
            if (std::abs(x_grad_data[i] - expected_x_grad[i]) > 1e-6) {
                std::cout << "✗ Gradient error for x at index " << i 
                          << ": expected " << expected_x_grad[i] 
                          << ", got " << x_grad_data[i] << std::endl;
                gradients_correct = false;
            }
        }
    } else {
        std::cout << "✗ No gradient computed for x" << std::endl;
        gradients_correct = false;
    }
    
    // Check gradient for y
    if (y.grad()) {
        y.grad()->realize();
        const auto& y_grad_data = y.grad()->data();
        
        std::cout << "Gradient for y (∂z/∂y): ";
        for (size_t i = 0; i < y_grad_data.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << y_grad_data[i];
        }
        std::cout << std::endl;
        
        // Expected: ∂z/∂y = x = [2, 3, 4]
        std::vector<float> expected_y_grad = {2.0f, 3.0f, 4.0f};
        for (size_t i = 0; i < y_grad_data.size(); ++i) {
            if (std::abs(y_grad_data[i] - expected_y_grad[i]) > 1e-6) {
                std::cout << "✗ Gradient error for y at index " << i 
                          << ": expected " << expected_y_grad[i] 
                          << ", got " << y_grad_data[i] << std::endl;
                gradients_correct = false;
            }
        }
    } else {
        std::cout << "✗ No gradient computed for y" << std::endl;
        gradients_correct = false;
    }
    
    if (gradients_correct) {
        std::cout << "✓ All gradients are correct!" << std::endl;
    } else {
        std::cout << "✗ Gradient computation failed!" << std::endl;
        return 1;
    }
    
    // Test more complex gradient scenario with chain rule
    std::cout << "\n=== Testing Chain Rule with Multiplication ===" << std::endl;
    
    // Create a more complex computation: w = (a * b) * c
    shao::Tensor<float> a({1.0f, 2.0f});
    shao::Tensor<float> b({3.0f, 4.0f});
    shao::Tensor<float> c({5.0f, 6.0f});
    
    // First multiplication: temp = a * b
    shao::Tensor<float> temp = mul_op(a, b);
    temp.realize();
    
    // Second multiplication: w = temp * c
    shao::Tensor<float> w = mul_op(temp, c);
    w.realize();
    
    std::cout << "a = [1, 2]" << std::endl;
    std::cout << "b = [3, 4]" << std::endl;
    std::cout << "c = [5, 6]" << std::endl;
    std::cout << "temp = a * b = [3, 8]" << std::endl;
    std::cout << "w = temp * c = [15, 48]" << std::endl;
    
    // Initialize gradient for output
    std::vector<float> w_ones(w.data().size(), 1.0f);
    w.set_grad(std::make_shared<shao::Tensor<float>>(w_ones, w.device()));
    
    // Run backward pass
    std::cout << "\nRunning backward pass for chain rule test..." << std::endl;
    w.backward();
    
    // Check gradients for chain rule
    // For w = (a * b) * c:
    // ∂w/∂a = ∂w/∂temp * ∂temp/∂a = c * b = [5, 6] * [3, 4] = [15, 24]
    // ∂w/∂b = ∂w/∂temp * ∂temp/∂b = c * a = [5, 6] * [1, 2] = [5, 12]
    // ∂w/∂c = temp = [3, 8]
    
    bool chain_rule_correct = true;
    
    // Check gradient for a
    if (a.grad()) {
        a.grad()->realize();
        const auto& a_grad_data = a.grad()->data();
        
        std::cout << "Gradient for a (∂w/∂a): ";
        for (size_t i = 0; i < a_grad_data.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << a_grad_data[i];
        }
        std::cout << std::endl;
        
        // Expected: ∂w/∂a = c * b = [5, 6] * [3, 4] = [15, 24]
        std::vector<float> expected_a_grad = {15.0f, 24.0f};
        for (size_t i = 0; i < a_grad_data.size(); ++i) {
            if (std::abs(a_grad_data[i] - expected_a_grad[i]) > 1e-6) {
                std::cout << "✗ Chain rule gradient error for a at index " << i 
                          << ": expected " << expected_a_grad[i] 
                          << ", got " << a_grad_data[i] << std::endl;
                chain_rule_correct = false;
            }
        }
    } else {
        std::cout << "✗ No gradient computed for a in chain rule test" << std::endl;
        chain_rule_correct = false;
    }
    
    // Check gradient for b
    if (b.grad()) {
        b.grad()->realize();
        const auto& b_grad_data = b.grad()->data();
        
        std::cout << "Gradient for b (∂w/∂b): ";
        for (size_t i = 0; i < b_grad_data.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << b_grad_data[i];
        }
        std::cout << std::endl;
        
        // Expected: ∂w/∂b = c * a = [5, 6] * [1, 2] = [5, 12]
        std::vector<float> expected_b_grad = {5.0f, 12.0f};
        for (size_t i = 0; i < b_grad_data.size(); ++i) {
            if (std::abs(b_grad_data[i] - expected_b_grad[i]) > 1e-6) {
                std::cout << "✗ Chain rule gradient error for b at index " << i 
                          << ": expected " << expected_b_grad[i] 
                          << ", got " << b_grad_data[i] << std::endl;
                chain_rule_correct = false;
            }
        }
    } else {
        std::cout << "✗ No gradient computed for b in chain rule test" << std::endl;
        chain_rule_correct = false;
    }
    
    // Check gradient for c
    if (c.grad()) {
        c.grad()->realize();
        const auto& c_grad_data = c.grad()->data();
        
        std::cout << "Gradient for c (∂w/∂c): ";
        for (size_t i = 0; i < c_grad_data.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << c_grad_data[i];
        }
        std::cout << std::endl;
        
        // Expected: ∂w/∂c = temp = [3, 8]
        std::vector<float> expected_c_grad = {3.0f, 8.0f};
        for (size_t i = 0; i < c_grad_data.size(); ++i) {
            if (std::abs(c_grad_data[i] - expected_c_grad[i]) > 1e-6) {
                std::cout << "✗ Chain rule gradient error for c at index " << i 
                          << ": expected " << expected_c_grad[i] 
                          << ", got " << c_grad_data[i] << std::endl;
                chain_rule_correct = false;
            }
        }
    } else {
        std::cout << "✗ No gradient computed for c in chain rule test" << std::endl;
        chain_rule_correct = false;
    }
    
    if (chain_rule_correct) {
        std::cout << "✓ Chain rule gradients are correct!" << std::endl;
    } else {
        std::cout << "✗ Chain rule gradient computation failed!" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✓ Forward computation: element-wise multiplication" << std::endl;
    std::cout << "✓ Partial derivatives: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a" << std::endl;
    std::cout << "✓ Backward computation: automatic differentiation" << std::endl;
    std::cout << "✓ Chain rule: complex computational graphs" << std::endl;
    std::cout << "✓ All multiplication operation tests passed!" << std::endl;
    return 0;
} 