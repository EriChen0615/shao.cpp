#pragma once

#include <vector>
#include <memory>
#include "tensor.hpp"

namespace shao {

// Forward declaration
template<typename T> class Tensor;

template<typename T>
class Op {
public:
    Op() = default;
    virtual std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) = 0;
    virtual void compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) = 0;
    
    // Automatic differentiation: compute partial derivative ∂to_tensor/∂from_tensor
    // Returns: partial_derivative = ∂to_tensor/∂from_tensor
    // from_tensor_list: all input tensors to the operation
    // target_tensor_id: ID of the tensor to differentiate with respect to
    virtual std::shared_ptr<Tensor<T>> partial_adjoint(const std::vector<Tensor<T>*>& from_tensor_list, Tensor<T>* to_tensor, size_t target_tensor_id) = 0;
    
    virtual ~Op() = default;

protected:
    // Validate that all input tensors are on the same device
    Device validate_device_consistency(const std::vector<Tensor<T>*>& inputs) const {
        if (inputs.empty()) {
            throw std::runtime_error("Op: No input tensors provided");
        }
        
        Device device = inputs[0]->device();
        for (size_t i = 1; i < inputs.size(); ++i) {
            if (inputs[i]->device() != device) {
                throw std::runtime_error("Op: All input tensors must be on the same device");
            }
        }
        return device;
    }
};

template<typename T>
class AddTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override;
    void compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) override;
    std::shared_ptr<Tensor<T>> partial_adjoint(const std::vector<Tensor<T>*>& from_tensor_list, Tensor<T>* to_tensor, size_t target_tensor_id) override;
    Tensor<T> operator()(Tensor<T>& a, Tensor<T>& b);
};

template<typename T>
class SumTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override;
    void compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) override;
    std::shared_ptr<Tensor<T>> partial_adjoint(const std::vector<Tensor<T>*>& from_tensor_list, Tensor<T>* to_tensor, size_t target_tensor_id) override;
    Tensor<T> operator()(const std::vector<Tensor<T>*>& tensors);
    Tensor<T> operator()(const std::vector<std::shared_ptr<Tensor<T>>>& tensors);
};

template<typename T>
class MulTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override;
    void compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) override;
    std::shared_ptr<Tensor<T>> partial_adjoint(const std::vector<Tensor<T>*>& from_tensor_list, Tensor<T>* to_tensor, size_t target_tensor_id) override;
    Tensor<T> operator()(Tensor<T>& a, Tensor<T>& b);
};

} // namespace shao
