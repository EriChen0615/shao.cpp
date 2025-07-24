#pragma once

#include <vector>
#include <memory>

namespace shao {

enum class Device {
    CPU,
    CUDA
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

    Device device() const { return device_; }
    
    void realize();
    
    // Move data to specified device
    void to_device(Device target_device);
    


protected:
    // Helper methods for CUDA memory management (accessible to ops)
    void allocate_cuda_memory();
    void free_cuda_memory();
    void copy_to_cuda();
    void copy_from_cuda();

private:
    std::shared_ptr<Op<T>> op_ = nullptr;
    std::vector<Tensor<T>*> inputs_;
    std::vector<T> cached_data_;
    Device device_ = Device::CPU;
    T* d_data_ptr_ = nullptr;  // CUDA memory pointer
    bool cached_data_is_stale_ = false;  // True if CPU cache is stale (CUDA data is newer)
    size_t size_ = 0;  // Size of the tensor data

    // Allow ops to construct tensors
    friend class AddTensorOp<T>;
};

} // namespace shao
