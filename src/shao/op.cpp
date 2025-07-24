#include "op.hpp"
#include "tensor.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <stdexcept>

extern "C" int cuda_add(float *h_a, float *h_b, float *h_c, int n);

namespace shao {

template<typename T>
std::vector<T> AddTensorOp<T>::compute(const std::vector<Tensor<T>*>& inputs) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    const auto& v1 = inputs[0]->data();
    const auto& v2 = inputs[1]->data();
    std::vector<T> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

template<typename T>
void AddTensorOp<T>::compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    
    if (device != Device::CUDA) {
        throw std::runtime_error("AddTensorOp::compute_cuda: Inputs must be on CUDA");
    }
    
    // For now, only support float for CUDA operations
    if constexpr (std::is_same_v<T, float>) {
        size_t size = inputs[0]->data().size();
        
        // Use CUDA data if available
        float *d_a = inputs[0]->d_data_ptr_;
        float *d_b = inputs[1]->d_data_ptr_;
        
        if (d_a && d_b && output_ptr) {
            // Run CUDA kernel into pre-allocated output memory
            cuda_add(d_a, d_b, output_ptr, size);
        } else {
            throw std::runtime_error("AddTensorOp::compute_cuda: CUDA data or output memory not available");
        }
    } else {
        throw std::runtime_error("AddTensorOp: CUDA operations only support float type");
    }
}

template<typename T>
Tensor<T> AddTensorOp<T>::operator()(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<AddTensorOp<T>>();
    new_tensor.inputs_.push_back(&a);
    new_tensor.inputs_.push_back(&b);
    // Set the device based on input tensors (they should be on the same device)
    new_tensor.device_ = a.device();
    
    // Set the size of the result tensor (same as input size for addition)
    new_tensor.size_ = a.data().size();
    
    return new_tensor;
}

// Explicit template instantiations
template class AddTensorOp<float>;
template class AddTensorOp<double>;

} // namespace shao 