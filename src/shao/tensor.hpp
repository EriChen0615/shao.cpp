#pragma once

#include <vector>
#include <memory>

namespace shao {

enum class Device {
    CPU,
    GPU
};

// Forward declarations
template<typename T> class Op;
template<typename T> class AddTensorOp;

template<typename T>
class Tensor {
public:
    Tensor(std::initializer_list<T> data, Device device = Device::CPU);
    Tensor(const std::vector<T>& data, Device device = Device::CPU);
    Tensor() = default;
    ~Tensor();

    const std::vector<T>& data() const {
        if (op_ && cached_data_.empty()) {
            const_cast<Tensor*>(this)->realize();
        }
        
        // If data is on GPU and CPU cache is stale, copy from GPU
        if (device_ == Device::GPU && d_data_ptr_ && cached_data_is_stale_) {
            const_cast<Tensor*>(this)->copy_from_gpu();
            const_cast<Tensor*>(this)->cached_data_is_stale_ = false;
        }
        
        return cached_data_;
    }

    Device device() const { return device_; }
    
    // GPU data access (only valid if device_ == Device::GPU)
    T* gpu_data() const { 
        if (device_ == Device::GPU && d_data_ptr_) {
            return d_data_ptr_; 
        }
        return nullptr;
    }

    void realize();
    
    // Move data to specified device
    void to_device(Device target_device);
    


protected:
    // Helper methods for GPU memory management (accessible to ops)
    void allocate_gpu_memory();
    void free_gpu_memory();
    void copy_to_gpu();
    void copy_from_gpu();

private:
    std::shared_ptr<Op<T>> op_ = nullptr;
    std::vector<Tensor<T>*> inputs_;
    std::vector<T> cached_data_;
    Device device_ = Device::CPU;
    T* d_data_ptr_ = nullptr;  // GPU memory pointer
    bool cached_data_is_stale_ = false;  // True if CPU cache is stale (GPU data is newer)
    size_t size_ = 0;  // Size of the tensor data

    // Allow ops to construct tensors
    friend class AddTensorOp<T>;
};

} // namespace shao
