#include "tensor.hpp"
#include "op.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <algorithm>

namespace shao {

template<typename T>
Tensor<T>::Tensor(std::initializer_list<T> data, Device device)
    : id_(next_id_++), cached_data_(data), device_(device), op_(nullptr), cached_data_is_stale_(false), size_(data.size()) {
    if (device_ == Device::CUDA) {
        allocate_cuda_memory();
        copy_to_cuda();
        cached_data_is_stale_ = true;  // CPU cache is now stale
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, Device device)
    : id_(next_id_++), cached_data_(data), device_(device), op_(nullptr), cached_data_is_stale_(false), size_(data.size()) {
    if (device_ == Device::CUDA) {
        allocate_cuda_memory();
        copy_to_cuda();
        cached_data_is_stale_ = true;  // CPU cache is now stale
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    if (device_ == Device::CUDA) {
        free_cuda_memory();
    }
}

// Default constructor implementation
template<typename T>
Tensor<T>::Tensor() : id_(next_id_++) {
}

template<typename T>
void Tensor<T>::print_info() const {
    std::cout << "Tensor ID: " << id_ 
              << ", Device: " << (device_ == Device::CPU ? "CPU" : "CUDA")
              << ", Size: " << size_
              << ", Has Op: " << (op_ != nullptr)
              << ", Input Count: " << inputs_.size();
    
    if (!inputs_.empty()) {
        std::cout << " (Input Tensor IDs: ";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << inputs_[i]->id();
        }
        std::cout << ")";
    }
    
    std::cout << ", Cached Data Size: " << cached_data_.size()
              << std::endl;
}

template<typename T>
void Tensor<T>::realize() {
    if (op_ && cached_data_.empty()) {
        // Compute all inputs first
        for (auto* input : inputs_) {
            input->realize();
        }
        
        if (device_ == Device::CUDA) {
            // CUDA computation - ensure we have memory allocated
            if (!d_data_ptr_) {
                allocate_cuda_memory();
            }
            
            // Tell operation to compute into our pre-allocated memory
            op_->compute_cuda(inputs_, d_data_ptr_);
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
    
    if (target_device == Device::CUDA) {
        // Moving to CUDA
        if (!d_data_ptr_) {
            allocate_cuda_memory();
        }
        copy_to_cuda();
        device_ = Device::CUDA;
    } else {
        // Moving to CPU
        if (d_data_ptr_) {
            copy_from_cuda();
            free_cuda_memory();
        }
        device_ = Device::CPU;
        cached_data_is_stale_ = false;  // CPU cache is now fresh
    }
}

template<typename T>
void Tensor<T>::allocate_cuda_memory() {
    if (size_ == 0) return;
    checkCudaErrors(cudaMalloc((void**)&d_data_ptr_, size_ * sizeof(T)));
}

template<typename T>
void Tensor<T>::free_cuda_memory() {
    if (d_data_ptr_) {
        checkCudaErrors(cudaFree(d_data_ptr_));
        d_data_ptr_ = nullptr;
    }
}

template<typename T>
void Tensor<T>::copy_to_cuda() {
    if (cached_data_.empty() || !d_data_ptr_ || size_ == 0) return;
    checkCudaErrors(cudaMemcpy(d_data_ptr_, cached_data_.data(), 
                               size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void Tensor<T>::copy_from_cuda() {
    if (!d_data_ptr_ || size_ == 0) return;
    
    // Resize cached_data_ if needed
    if (cached_data_.size() != size_) {
        cached_data_.resize(size_);
    }
    
    checkCudaErrors(cudaMemcpy(cached_data_.data(), d_data_ptr_, 
                               size_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::multiply_tensors(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
    MulTensorOp<T> mul_op;
    auto result = mul_op(*a, *b);
    return std::make_shared<Tensor<T>>(result);
}

template<typename T>
std::vector<Tensor<T>*> Tensor<T>::reverse_topo_sort(Tensor<T>* output_tensor) {
    std::vector<Tensor<T>*> topo_order;
    std::unordered_set<size_t> visited;
    
    // Helper function for DFS
    std::function<void(Tensor<T>*)> dfs = [&](Tensor<T>* tensor) {
        if (visited.find(tensor->id()) != visited.end()) {
            return;
        }
        visited.insert(tensor->id());
        
        // Visit all inputs first (post-order DFS)
        for (auto* input : tensor->inputs_) {
            dfs(input);
        }
        
        // Add current tensor after all its inputs
        topo_order.push_back(tensor);
    };
    
    // Start DFS from output tensor
    dfs(output_tensor);
    
    // Reverse to get reverse topological order
    std::reverse(topo_order.begin(), topo_order.end());
    
    return topo_order;
}

template<typename T>
void Tensor<T>::backward() {
    // TODO: Implement reverse automatic differentiation
    auto tensor_list = reverse_topo_sort(this);
    std::unordered_map<size_t, std::vector<std::shared_ptr<Tensor<T>>>> id_to_adjoints_map;
    SumTensorOp<T> sum_op;
    for (auto cur_tensor : tensor_list) { // For each tensor in reversed topological order

        // Step 1. Compute gradient of the tensor by summing partial adjoints. See https://dlsyscourse.org/slides/4-automatic-differentiation.pdf for the maths.
        auto adjoint_list = id_to_adjoints_map[cur_tensor->id()];
        if (!adjoint_list.empty()) {
            // Use shared pointers directly with sum_op
            auto sum_result = sum_op(adjoint_list);
            cur_tensor->grad_ = std::make_shared<Tensor<T>>(sum_result);
            cur_tensor->grad_->realize();
            for (auto adjoint : adjoint_list) {
                adjoint->print_info();
                std::cout << "Adjoint data: ";
                for (auto val : adjoint->data()) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }

        // Step 2. Compute partial adjoints for all input tensors
        for (auto input_tensor : cur_tensor->inputs_) {
            auto partial_adjoint = cur_tensor->op_->partial_adjoint(input_tensor, cur_tensor); // compute partial derivative
            // TODO: Multiply partial_derivative by cur_tensor->grad_ to get the full adjoint
            id_to_adjoints_map[input_tensor->id()].push_back(partial_adjoint);
        }
    }
}

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace shao
