//
//  QNNStridedSlice.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNSTRIDEDSLICE_HPP
#define MNN_QNNSTRIDEDSLICE_HPP

#include "QNNCommonExecution.hpp"

namespace MNN {
namespace QNN {

class QNNStridedSlice : public QNNCommonExecution {
public:
    QNNStridedSlice(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    uint32_t computeMask(uint32_t rawMask, int dim, Tensor::DimensionType dimType);
};

} // end namespace QNN
} // end namespace MNN

#endif // end MNN_QNNSTRIDEDSLICE_HPP
