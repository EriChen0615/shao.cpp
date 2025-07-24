#pragma once

#include <vector>
#include <memory>

namespace shao {

// Forward declarations
template<typename T> class Op;
template<typename T> class AddTensorOp;

template<typename T>
class Tensor {
public:
    Tensor(std::initializer_list<T> data);
    Tensor(const std::vector<T>& data);
    Tensor() = default;

    const std::vector<T>& data() const {
        if (op_ && cached_data_.empty()) {
            const_cast<Tensor*>(this)->realize();
        }
        return cached_data_;
    }

    void realize();

private:
    std::shared_ptr<Op<T>> op_ = nullptr;
    std::vector<Tensor<T>*> inputs_;
    std::vector<T> cached_data_;

    // Allow ops to construct tensors
    friend class AddTensorOp<T>;
};

} // namespace shao
