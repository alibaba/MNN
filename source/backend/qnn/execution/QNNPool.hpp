//
//  QNNPool.hpp
//  MNN
//
//  Created by MNN on b'2025/04/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_QNNPOOL_HPP
#define MNN_QNNPOOL_HPP

#include "QNNCommonExecution.hpp"
#include "QnnTypes.h"

namespace MNN {
namespace QNN {

class QNNPool : public QNNCommonExecution {
public:
    QNNPool(Backend *backend, const Op *op) : QNNCommonExecution(backend, op) {}
    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode onEncode3D(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

private:
    void setParamPool(std::string & nodeType, std::vector<uint32_t> & filterSizeData, std::vector<uint32_t> & strideData, std::vector<uint32_t> & padAmountData, uint32_t & roundingMode, Tensor * input, Tensor * output);
};

} // end namespace QNN
} // end namespace MNN


#endif
