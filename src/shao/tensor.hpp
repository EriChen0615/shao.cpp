#pragma once

#include <vector>
#include <memory>
#include <atomic>

namespace shao {

enum class Device {
    CPU,
    CUDA
};

// Forward declarations
template<typename T> class Op;
template<typename T> class AddTensorOp;
template<typename T> class SumTensorOp;
template<typename T> class MulTensorOp;

template<typename T>
class Tensor {
public:
    Tensor(std::initializer_list<T> data, Device device = Device::CPU);
    Tensor(const std::vector<T>& data, Device device = Device::CPU);
    Tensor();
    ~Tensor();

    std::vector<T>& data() {
        if (op_ && cached_data_.empty()) {
            realize();
        }
        
        // If data is on CUDA and CPU cache is stale, copy from CUDA
        if (device_ == Device::CUDA && d_data_ptr_ && cached_data_is_stale_) {
            copy_from_cuda();
            cached_data_is_stale_ = false;
        }
        
        return cached_data_;
    }
    
    const std::vector<T>& data() const {
        // For const access, we need to ensure data is realized
        if (op_ && cached_data_.empty()) {
            const_cast<Tensor<T>*>(this)->realize();
        }
        
        // If data is on CUDA and CPU cache is stale, copy from CUDA
        if (device_ == Device::CUDA && d_data_ptr_ && cached_data_is_stale_) {
            const_cast<Tensor<T>*>(this)->copy_from_cuda();
            const_cast<Tensor<T>*>(this)->cached_data_is_stale_ = false;
        }
        
        return cached_data_;
    }

    Device device() const { return device_; }
    
    // Get unique tensor ID
    size_t id() const { return id_; }
    
    // Debug method to print tensor information
    void print_info() const;
    
    // Static method to sum multiple tensors
    static Tensor<T> sum(const std::vector<Tensor<T>*>& tensors);
    
    // Static method to multiply two tensors (for chain rule)
    static std::shared_ptr<Tensor<T>> multiply_tensors(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b);
    
    // Helper functions for automatic differentiation
    static std::vector<Tensor<T>*> reverse_topo_sort(Tensor<T>* output_tensor);
    
    // Gradient accessors
    std::shared_ptr<Tensor<T>> grad() const { return grad_; }
    void set_grad(std::shared_ptr<Tensor<T>> grad) { grad_ = grad; }
    
    void realize();
    void backward();
    
    // Move data to specified device
    void to_device(Device target_device);
    


protected:
    // Helper methods for CUDA memory management (accessible to ops)
    void allocate_cuda_memory();
    void free_cuda_memory();
    void copy_to_cuda();
    void copy_from_cuda();

private:
    static std::atomic<size_t> next_id_;  // Static counter for unique IDs
    
    size_t id_;  // Unique tensor identifier
    std::shared_ptr<Op<T>> op_ = nullptr;
    std::vector<Tensor<T>*> inputs_;
    std::vector<T> cached_data_;
    std::shared_ptr<Tensor<T>> grad_ = nullptr;
    Device device_ = Device::CPU;
    T* d_data_ptr_ = nullptr;  // CUDA memory pointer
    bool cached_data_is_stale_ = false;  // True if CPU cache is stale (CUDA data is newer)
    size_t size_ = 0;  // Size of the tensor data

    // Allow ops to construct tensors
    friend class AddTensorOp<T>;
    friend class SumTensorOp<T>;
    friend class MulTensorOp<T>;
};

// Static member initialization
template<typename T>
std::atomic<size_t> Tensor<T>::next_id_{0};

} // namespace shao
