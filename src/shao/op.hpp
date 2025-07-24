#pragma once

#include <vector>
#include <memory>

namespace shao {

enum class Device {
    CPU,
    GPU
};
// Forward declaration
template<typename T> class Tensor;

template<typename T>
class Op {
public:
    Op(Device device=Device::CPU) : device_(device) {}
    virtual std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) = 0;
    virtual ~Op() = default;

private:
    Device device_;
};

template<typename T>
class AddTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override;
    Tensor<T> operator()(Tensor<T>& a, Tensor<T>& b);
};

} // namespace shao
