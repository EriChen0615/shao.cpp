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
std::shared_ptr<Tensor<T>> AddTensorOp<T>::partial_adjoint(Tensor<T>* from_tensor, Tensor<T>* to_tensor) {
    // For addition: ∂(a + b)/∂a = 1, ∂(a + b)/∂b = 1
    // The partial derivative is 1 for any input
    
    // Create a tensor of ones as the partial derivative
    std::vector<T> ones(to_tensor->data().size(), T(1));
    auto partial_derivative = std::make_shared<Tensor<T>>(ones, to_tensor->device());
    
    return partial_derivative;
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

template<typename T>
std::vector<T> SumTensorOp<T>::compute(const std::vector<Tensor<T>*>& inputs) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    
    if (inputs.empty()) {
        throw std::runtime_error("SumTensorOp::compute: No input tensors provided");
    }
    
    // Get the size from the first tensor
    size_t size = inputs[0]->data().size();
    std::vector<T> result(size, T(0));  // Initialize with zeros
    
    // Sum all input tensors
    for (const auto* input : inputs) {
        const auto& input_data = input->data();
        if (input_data.size() != size) {
            throw std::runtime_error("SumTensorOp::compute: All input tensors must have the same size");
        }
        
        for (size_t i = 0; i < size; ++i) {
            result[i] += input_data[i];
        }
    }
    
    return result;
}

template<typename T>
void SumTensorOp<T>::compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) {
    //TODO
}

template<typename T>
std::shared_ptr<Tensor<T>> SumTensorOp<T>::partial_adjoint(Tensor<T>* from_tensor, Tensor<T>* to_tensor) {
    // For sum: ∂(sum)/∂input = 1 for each input
    // The partial derivative is 1 for any input
    
    // Create a tensor of ones as the partial derivative
    std::vector<T> ones(to_tensor->data().size(), T(1));
    auto partial_derivative = std::make_shared<Tensor<T>>(ones, to_tensor->device());
    
    return partial_derivative;
}

template<typename T>
Tensor<T> SumTensorOp<T>::operator()(const std::vector<Tensor<T>*>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("SumTensorOp::operator(): No input tensors provided");
    }
    
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<SumTensorOp<T>>();
    
    // Add all input tensors to the inputs list
    for (const auto* tensor : tensors) {
        new_tensor.inputs_.push_back(const_cast<Tensor<T>*>(tensor));
    }
    
    // Set the device based on input tensors (they should be on the same device)
    new_tensor.device_ = tensors[0]->device();
    
    // Set the size of the result tensor (same as input size for summation)
    new_tensor.size_ = tensors[0]->data().size();
    
    return new_tensor;
}

template<typename T>
Tensor<T> SumTensorOp<T>::operator()(const std::vector<std::shared_ptr<Tensor<T>>>& tensors) {
    if (tensors.empty()) {
        throw std::runtime_error("SumTensorOp::operator(): No input tensors provided");
    }
    
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<SumTensorOp<T>>();
    
    // Add all input tensors to the inputs list
    for (const auto& tensor : tensors) {
        new_tensor.inputs_.push_back(tensor.get());
    }
    
    // Set the device based on input tensors (they should be on the same device)
    new_tensor.device_ = tensors[0]->device();
    
    // Set the size of the result tensor (same as input size for summation)
    new_tensor.size_ = tensors[0]->data().size();
    
    return new_tensor;
}

template<typename T>
std::vector<T> MulTensorOp<T>::compute(const std::vector<Tensor<T>*>& inputs) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    const auto& v1 = inputs[0]->data();
    const auto& v2 = inputs[1]->data();
    std::vector<T> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] * v2[i];
    }
    return result;
}

template<typename T>
void MulTensorOp<T>::compute_cuda(const std::vector<Tensor<T>*>& inputs, T* output_ptr) {
    // Validate that all inputs are on the same device
    Device device = this->validate_device_consistency(inputs);
    
    if (device != Device::CUDA) {
        throw std::runtime_error("MulTensorOp::compute_cuda: Inputs must be on CUDA");
    }
    
    // For now, only support float for CUDA operations
    if constexpr (std::is_same_v<T, float>) {
        size_t size = inputs[0]->data().size();
        
        // Use CUDA data if available
        float *d_a = inputs[0]->d_data_ptr_;
        float *d_b = inputs[1]->d_data_ptr_;
        
        if (d_a && d_b && output_ptr) {
            // TODO: Implement CUDA kernel for multiplication
            // For now, copy to host, compute, copy back
            std::vector<float> host_a(size);
            std::vector<float> host_b(size);
            checkCudaErrors(cudaMemcpy(host_a.data(), d_a, size * sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(host_b.data(), d_b, size * sizeof(float), cudaMemcpyDeviceToHost));
            
            std::vector<float> host_result(size);
            for (size_t i = 0; i < size; ++i) {
                host_result[i] = host_a[i] * host_b[i];
            }
            
            checkCudaErrors(cudaMemcpy(output_ptr, host_result.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            throw std::runtime_error("MulTensorOp::compute_cuda: CUDA data or output memory not available");
        }
    } else {
        throw std::runtime_error("MulTensorOp: CUDA operations only support float type");
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> MulTensorOp<T>::partial_adjoint(Tensor<T>* from_tensor, Tensor<T>* to_tensor) {
    // For multiplication: ∂(a * b)/∂a = b, ∂(a * b)/∂b = a
    // The partial derivative depends on which input we're differentiating with respect to
    
    // Find the other input tensor
    Tensor<T>* other_tensor = nullptr;
    for (auto* input : to_tensor->inputs_) {
        if (input != from_tensor) {
            other_tensor = input;
            break;
        }
    }
    
    if (!other_tensor) {
        throw std::runtime_error("MulTensorOp::partial_adjoint: Could not find other input tensor");
    }
    
    // Return the other tensor as the partial derivative
    const auto& other_data = other_tensor->data();
    auto partial_derivative = std::make_shared<Tensor<T>>(other_data, to_tensor->device());
    
    return partial_derivative;
}

template<typename T>
Tensor<T> MulTensorOp<T>::operator()(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<MulTensorOp<T>>();
    new_tensor.inputs_.push_back(&a);
    new_tensor.inputs_.push_back(&b);
    // Set the device based on input tensors (they should be on the same device)
    new_tensor.device_ = a.device();
    
    // Set the size of the result tensor (same as input size for multiplication)
    new_tensor.size_ = a.data().size();
    
    return new_tensor;
}

// Explicit template instantiations
template class AddTensorOp<float>;
template class AddTensorOp<double>;
template class SumTensorOp<float>;
template class SumTensorOp<double>;
template class MulTensorOp<float>;
template class MulTensorOp<double>;

} // namespace shao 