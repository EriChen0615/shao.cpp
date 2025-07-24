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
T* AddTensorOp<T>::compute_gpu(const std::vector<Tensor<T>*>& inputs, size_t& size) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    
    if (device != Device::GPU) {
        throw std::runtime_error("AddTensorOp::compute_gpu: Inputs must be on GPU");
    }
    
    // For now, only support float for GPU operations
    if constexpr (std::is_same_v<T, float>) {
        size = inputs[0]->data().size();
        
        // Use GPU data if available
        float *d_a = inputs[0]->gpu_data();
        float *d_b = inputs[1]->gpu_data();
        
        if (d_a && d_b) {
            // Allocate GPU memory for result
            float *d_res;
            checkCudaErrors(cudaMalloc((void**)&d_res, size * sizeof(float)));
            
            // Run GPU kernel
            cuda_add(d_a, d_b, d_res, size);
            
            return d_res;  // Return the GPU pointer
        } else {
            throw std::runtime_error("AddTensorOp::compute_gpu: GPU data not available");
        }
    } else {
        throw std::runtime_error("AddTensorOp: GPU operations only support float type");
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
    
    // If inputs are on GPU, pre-allocate GPU memory for the result
    if (new_tensor.device_ == Device::GPU) {
        new_tensor.allocate_gpu_memory();
    }
    
    return new_tensor;
}

// Explicit template instantiations
template class AddTensorOp<float>;
template class AddTensorOp<double>;

} // namespace shao 