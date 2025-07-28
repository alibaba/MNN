#ifndef MNN_QNNAttention_HPP
#define MNN_QNNAttention_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNAttention : public QNNCommonExecution {
public:
    QNNAttention(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNAttention_HPP
