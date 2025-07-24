#include "tensor.hpp"
#include "op.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <stdexcept>

namespace shao {

template<typename T>
Tensor<T>::Tensor(std::initializer_list<T> data, Device device)
    : cached_data_(data), device_(device), op_(nullptr), cached_data_is_stale_(false), size_(data.size()) {
    if (device_ == Device::GPU) {
        allocate_gpu_memory();
        copy_to_gpu();
        cached_data_is_stale_ = true;  // CPU cache is now stale
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, Device device)
    : cached_data_(data), device_(device), op_(nullptr), cached_data_is_stale_(false), size_(data.size()) {
    if (device_ == Device::GPU) {
        allocate_gpu_memory();
        copy_to_gpu();
        cached_data_is_stale_ = true;  // CPU cache is now stale
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    if (device_ == Device::GPU) {
        free_gpu_memory();
    }
}

template<typename T>
void Tensor<T>::realize() {
    if (op_ && cached_data_.empty()) {
        // Compute all inputs first
        for (auto* input : inputs_) {
            input->realize();
        }
        
        if (device_ == Device::GPU) {
            // GPU computation - compute directly to GPU memory
            if (d_data_ptr_) {
                free_gpu_memory(); // Free old memory before re-allocating
            }
            size_t result_size;
            d_data_ptr_ = op_->compute_gpu(inputs_, result_size);
            size_ = result_size;  // Store the result size
            cached_data_is_stale_ = true;  // CPU cache is now stale
        } else {
            // CPU computation
            cached_data_ = op_->compute(inputs_);
            size_ = cached_data_.size();  // Store the result size
        }
    }
}

template<typename T>
void Tensor<T>::to_device(Device target_device) {
    if (device_ == target_device) return;
    
    if (target_device == Device::GPU) {
        // Moving to GPU
        if (!d_data_ptr_) {
            allocate_gpu_memory();
        }
        copy_to_gpu();
        device_ = Device::GPU;
    } else {
        // Moving to CPU
        if (d_data_ptr_) {
            copy_from_gpu();
            free_gpu_memory();
        }
        device_ = Device::CPU;
        cached_data_is_stale_ = false;  // CPU cache is now fresh
    }
}

template<typename T>
void Tensor<T>::allocate_gpu_memory() {
    if (size_ == 0) return;
    checkCudaErrors(cudaMalloc((void**)&d_data_ptr_, size_ * sizeof(T)));
}

template<typename T>
void Tensor<T>::free_gpu_memory() {
    if (d_data_ptr_) {
        checkCudaErrors(cudaFree(d_data_ptr_));
        d_data_ptr_ = nullptr;
    }
}

template<typename T>
void Tensor<T>::copy_to_gpu() {
    if (cached_data_.empty() || !d_data_ptr_ || size_ == 0) return;
    checkCudaErrors(cudaMemcpy(d_data_ptr_, cached_data_.data(), 
                               size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void Tensor<T>::copy_from_gpu() {
    if (!d_data_ptr_ || size_ == 0) return;
    
    // Resize cached_data_ if needed
    if (cached_data_.size() != size_) {
        cached_data_.resize(size_);
    }
    
    checkCudaErrors(cudaMemcpy(cached_data_.data(), d_data_ptr_, 
                               size_ * sizeof(T), cudaMemcpyDeviceToHost));
}



// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace shao
