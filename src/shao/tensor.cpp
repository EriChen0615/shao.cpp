#include "tensor.hpp"
#include "op.hpp"

namespace shao {

template<typename T>
Tensor<T>::Tensor(std::initializer_list<T> data) : cached_data_(data), op_(nullptr) {}

template<typename T>
void Tensor<T>::realize() {
    if (op_ && cached_data_.empty()) {
        cached_data_ = op_->compute(inputs_);
    }
}

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace shao
