#ifndef MNN_QNNPERMUTE_HPP
#define MNN_QNNPERMUTE_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNPermute : public QNNCommonExecution {
public:
    QNNPermute(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNPERMUTE_HPP
