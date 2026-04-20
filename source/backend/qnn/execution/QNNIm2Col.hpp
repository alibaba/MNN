//
//  QNNIm2Col.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNIM2COL_HPP
#define MNN_QNNIM2COL_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNIm2Col : public QNNCommonExecution {
public:
    QNNIm2Col(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif
