//
//  QNNBinary.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNBINARY_HPP
#define MNN_QNNBINARY_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

class QNNBinary : public QNNCommonExecution {
public:
    QNNBinary(Backend *backend, const Op *op, const std::string & binaryTypeName) : QNNCommonExecution(backend, op), mBinaryTypeName(binaryTypeName) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    ErrorCode onEncodeScalarOptimize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int fullIndex);
    ErrorCode onEncodeBroadcast(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, int fullIndex);
private:
    std::string mBinaryTypeName;
};
#endif
} // end namespace QNN
} // end namespace MNN

#endif // end MNN_BINARY_HPP

