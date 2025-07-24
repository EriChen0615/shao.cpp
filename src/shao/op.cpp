#include "op.hpp"
#include "tensor.hpp"

namespace shao {

template<typename T>
std::vector<T> AddTensorOp<T>::compute(const std::vector<Tensor<T>*>& inputs) {
    if (device_ == Device::CPU) {
    const auto& v1 = inputs[0]->data();
    const auto& v2 = inputs[1]->data();
    std::vector<T> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    } else if (device_ == Device::GPU) {
        throw std::runtime_error("AddTensorOp: GPU not supported yet");
    } else {
        throw std::runtime_error("AddTensorOp: Invalid device");
    }
}

template<typename T>
Tensor<T> AddTensorOp<T>::operator()(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> new_tensor;
    new_tensor.op_ = std::make_shared<AddTensorOp<T>>();
    new_tensor.inputs_.push_back(&a);
    new_tensor.inputs_.push_back(&b);
    return new_tensor;
}

// Explicit template instantiations
template class AddTensorOp<float>;
template class AddTensorOp<double>;

} // namespace shao 