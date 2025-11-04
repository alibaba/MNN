#ifndef MNN_QNNMATMUL_HPP
#define MNN_QNNMATMUL_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNMatMul : public QNNCommonExecution {
public:
    QNNMatMul(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNMATMUL_HPP
