//
//  CPURasterDiff.cpp
//  MNN
//
//  Created by MNN on 2023/07/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "../CPUBackend.hpp"
#include "../compute/CommonOptFunction.h"

namespace MNN {
#ifdef MNN_SUPPORT_RENDER

class CPURasterDiff : public Execution {
public:
    CPURasterDiff(Backend* bn) : Execution(bn) {
        // Do nothing
    }
    virtual ~ CPURasterDiff() {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto width = inputs[0]->length(2);
        auto height = inputs[0]->length(1);
        auto batch = inputs[0]->length(0);
        auto unit = inputs[0]->length(3);
        for (int b=0; b<batch; ++b) {
            auto dxPtr = outputs[0]->host<float>() + b * width * height * unit;
            auto dyPtr = outputs[1]->host<float>() + b * width * height * unit;
            auto inputPtr = inputs[0]->host<const float>() + b * width * height * unit;
            // Compute Dx
            for (int y=0; y<height; ++y) {
                auto dxPtrY = dxPtr + y * width * unit;
                auto inputPtrY = inputPtr + y * width * unit;
                for (int x=0; x<width-1; ++x) {
                    auto dxPtrX = dxPtrY + x * unit;
                    auto dyPtrX = dxPtrY + x * unit;
                    auto iPtrX = inputPtrY+ (x+1) * unit;
                    auto iPtrXO = inputPtrY + (x) * unit;
                    for (int c=0; c<unit; ++c) {
                        dxPtrX[c] = iPtrX[c] - iPtrXO[c];
                    }
                }
                // Last X is zero
                for (int c=0; c<unit; ++c) {
                    dxPtrY[(width-1)*unit+c] = 0.0f;
                }
            }
            // Compute DY
            for (int y=0; y<height-1; ++y) {
                auto dyPtrY = dyPtr + y * width * unit;
                auto inputPtrY = inputPtr + (y+1) * width * unit;
                auto inputPtrYO = inputPtr + y * width * unit;
                for (int x=0; x<width; ++x) {
                    auto dyPtrX = dyPtrY + x * unit;
                    auto iPtrX = inputPtrY+ x * unit;
                    auto iPtrXO = inputPtrYO + x * unit;
                    for (int c=0; c<unit; ++c) {
                        dyPtrX[c] = iPtrX[c] - iPtrXO[c];
                    }
                }
            }
            ::memset(dyPtr + (height-1)*width*unit, 0, width*unit*sizeof(float));
        }
        return NO_ERROR;
    }
};

class CPURasterDiffGrad : public Execution {
public:
    CPURasterDiffGrad(Backend* bn) : Execution(bn) {
        // Do nothing
    }
    virtual ~ CPURasterDiffGrad() {
        // Do nothing
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        MNN_ASSERT(inputs.size() == 2);
        MNN_ASSERT(outputs.size() == 1);
        auto width = inputs[0]->length(2);
        auto height = inputs[0]->length(1);
        auto batch = inputs[0]->length(0);
        auto unit = inputs[0]->length(3);
        ::memset(outputs[0]->host<float>(), 0, width * height * batch * unit * sizeof(float));
        for (int b=0; b<batch; ++b) {
            auto dxPtr = inputs[0]->host<const float>() + b * width * height * unit;
            auto dyPtr = inputs[1]->host<const float>() + b * width * height * unit;
            auto inputPtr = outputs[0]->host<float>() + b * width * height * unit;
            // Compute Dx
            for (int y=0; y<height; ++y) {
                auto dxPtrY = dxPtr + y * width * unit;
                auto inputPtrY = inputPtr + y * width * unit;
                for (int x=0; x<width-1; ++x) {
                    auto dxPtrX = dxPtrY + x * unit;
                    auto dyPtrX = dxPtrY + x * unit;
                    auto iPtrX = inputPtrY+ (x+1) * unit;
                    auto iPtrXO = inputPtrY + (x) * unit;
                    for (int c=0; c<unit; ++c) {
                        iPtrX[c] += dxPtrX[c];
                        iPtrXO[c] -= dxPtrX[c];
                    }
                }
            }
            // Compute DY
            for (int y=0; y<height-1; ++y) {
                auto dyPtrY = dyPtr + y * width * unit;
                auto inputPtrY = inputPtr + (y+1) * width * unit;
                auto inputPtrYO = inputPtr + y * width * unit;
                for (int x=0; x<width; ++x) {
                    auto dyPtrX = dyPtrY + x * unit;
                    auto iPtrX = inputPtrY+ x * unit;
                    auto iPtrXO = inputPtrYO + x * unit;
                    for (int c=0; c<unit; ++c) {
                        iPtrX[c] += dyPtrY[c];
                        iPtrXO[c] -= dyPtrY[c];
                    }
                }
            }
        }
        return NO_ERROR;
    }
};

class CPURasterDiffCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        if (nullptr != op->main_as_Extra()) {
            return new CPURasterDiffGrad(backend);
        }
        return new CPURasterDiff(backend);
    }
};
#endif

REGISTER_CPU_OP_CREATOR_RENDER(CPURasterDiffCreator, OpType_RasterDiff);

} // namespace MNN
