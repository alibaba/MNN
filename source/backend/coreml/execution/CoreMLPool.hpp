//
//  CoreMLPool.hpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLPOOL_HPP
#define MNN_COREMLPOOL_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLPool : public CoreMLCommonExecution {
public:
    CoreMLPool(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLPool() = default;
private:
    void addPadLayer(const Tensor * input, const Pool* common);
    std::string mPoolInputName, mPoolOutputName;
};
} // namespace MNN

#endif // MNN_COREMLPOOL_HPP
