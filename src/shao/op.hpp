#pragma once

#include <vector>
#include <memory>

namespace shao {

// Forward declaration
template<typename T> class Tensor;

template<typename T>
class Op {
public:
    Op() = default;
    virtual std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) = 0;
    virtual ~Op() = default;
};

template<typename T>
class AddTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override;
    Tensor<T> operator()(Tensor<T>& a, Tensor<T>& b);
};

} // namespace shao
