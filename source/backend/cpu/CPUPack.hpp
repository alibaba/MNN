//
//  CPUPack.hpp
//  MNN
//
//  Created by MNN on 2018/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPack_hpp
#define CPUPack_hpp

#include "Execution.hpp"
#include "Type_generated.h"

namespace MNN {
class CPUPack : public Execution {
public:
    CPUPack(Backend *backend, const Op *op, DataType type, int axis);
    virtual ~CPUPack() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    template <typename T>
    ErrorCode MNNPackLayerForward(const std::vector<MNN::Tensor *> &inputs, const std::vector<MNN::Tensor *> &outputs);

private:
    DataType mDataType;
    int mAxis;
};
} // namespace MNN
#endif /* CPUPack_hpp */
