#include <iostream>
#include <initializer_list>
#include <vector>
#include <memory>

namespace shao {

// Forward declarations
template<typename T> class Tensor;
template<typename T> class AddTensorOp;

template<typename T>
class Op {
public:
    Op() = default;
    virtual std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) = 0;
    virtual ~Op() = default;
};

template<typename T>
class Tensor {
public:
    Tensor(std::initializer_list<T> data) : cached_data_(data), op_(nullptr) {}
    Tensor() = default;

    const std::vector<T>& data() const { return cached_data_; }
    std::vector<T>& mutable_data() { return cached_data_; }

    void realize() {
        if (op_ && cached_data_.empty()) {
            cached_data_ = op_->compute(inputs_);
        }
    }

private:
    std::shared_ptr<Op<T>> op_ = nullptr;
    std::vector<Tensor<T>*> inputs_;
    std::vector<T> cached_data_;

    // Allow ops to construct tensors
    friend class AddTensorOp<T>;
};

template<typename T>
class AddTensorOp : public Op<T> {
public:
    std::vector<T> compute(const std::vector<Tensor<T>*>& inputs) override {
        const auto& v1 = inputs[0]->data();
        const auto& v2 = inputs[1]->data();
        std::vector<T> result(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    }

    Tensor<T> operator()(Tensor<T>& a, Tensor<T>& b) {
        Tensor<T> new_tensor;
        new_tensor.op_ = std::make_shared<AddTensorOp<T>>();
        new_tensor.inputs_.push_back(&a);
        new_tensor.inputs_.push_back(&b);
        new_tensor.realize();  // eager execution
        return new_tensor;
    }
};

}  // namespace shao

int main() {
    shao::Tensor<float> a({1, 2, 3, 4});
    shao::Tensor<float> b({2, 4, 6, 8});

    shao::AddTensorOp<float> add_op;
    auto c = add_op(a, b);

    std::cout << "c: ";
    for (auto x : c.data()) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}