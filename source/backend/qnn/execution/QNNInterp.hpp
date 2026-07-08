//
//  QNNInterp.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNINTERP_HPP
#define MNN_QNNINTERP_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNInterp : public QNNCommonExecution {
public:
    QNNInterp(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
