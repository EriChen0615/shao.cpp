#include "op.hpp"
#include "tensor.hpp"
#include "helper_cuda.h"
#include <stdexcept>

extern "C" int cuda_add(float *h_a, float *h_b, float *h_c, int n);

namespace shao {

template<typename T>
std::vector<T> AddTensorOp<T>::compute(const std::vector<Tensor<T>*>& inputs) {
    if (this->device_ == Device::CPU) {
        const auto& v1 = inputs[0]->data();
        const auto& v2 = inputs[1]->data();
        std::vector<T> result(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    } else if (this->device_ == Device::GPU) {
        // For now, only support float for GPU operations
        if constexpr (std::is_same_v<T, float>) {
            const auto& v1 = inputs[0]->data();
            const auto& v2 = inputs[1]->data();
            std::vector<T> result(v1.size());
            cuda_add(
                const_cast<float*>(v1.data()),
                const_cast<float*>(v2.data()),
                result.data(),
                v1.size()
            );
            return result;
        } else {
            throw std::runtime_error("AddTensorOp: GPU operations only support float type");
        }
    } else {
        throw std::runtime_error("AddTensorOp: Invalid device");
    }
}

template<typename T>
Tensor<T> AddTensorOp<T>::operator()(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<AddTensorOp<T>>(this->device_);
    new_tensor.inputs_.push_back(&a);
    new_tensor.inputs_.push_back(&b);
    return new_tensor;
}

// Explicit template instantiations
template class AddTensorOp<float>;
template class AddTensorOp<double>;

} // namespace shao 