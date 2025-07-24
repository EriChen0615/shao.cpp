#include <iostream>
#include "shao/tensor.hpp"
#include "shao/op.hpp"

int main() {
    shao::Tensor<float> a({1, 2, 3, 4});
    shao::Tensor<float> b({2, 4, 6, 8});

    shao::AddTensorOp<float> add_op;
    auto c = add_op(a, b, shao::Device::GPU);

    std::cout << "c: ";
    for (auto x : c.data()) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
} 