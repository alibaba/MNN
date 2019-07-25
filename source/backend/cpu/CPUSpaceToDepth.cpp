//
//  CPUSpaceToDepth.hpp
//  MNN
//
//  Created by MNN on 2019/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSpaceToDepth.hpp"
#include "Backend.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUSpaceToDepth<T>::CPUSpaceToDepth(Backend* backend, const MNN::Op* op) : Execution(backend), mOp(op) {
    // do nothing
}

template <typename T>
ErrorCode CPUSpaceToDepth<T>::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    // do nothing

    return NO_ERROR;
}

template <typename T>
ErrorCode CPUSpaceToDepth<T>::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const int blockSize = mOp->main_as_DepthSpaceParam()->blockSize();

    const int inputBatch = ib.dim[0].extent;
    const int inputHeight = ib.dim[1].extent;
    const int inputWidth = ib.dim[2].extent;
    const int inputChannels = ib.dim[3].extent;

    const int outputHeight = ob.dim[1].extent;
    const int outputWidth = ob.dim[2].extent;
    const int outputChannels = ob.dim[3].extent;

    T* inputOrigin = reinterpret_cast<T*>(ib.host);
    T* outputDest = reinterpret_cast<T*>(ob.host);

    // NHWC
    // TODO: implement NC4HW4
    for (int b = 0; b < inputBatch; b++) {
        for (int h = 0; h < inputHeight; h++) {
            const int oh = h / blockSize;
            const int offsetH = h % blockSize;
            for (int w = 0; w < inputWidth; w++) {
                const int ow = w / blockSize;
                const int offsetW = w % blockSize;
                const int offsetC = (offsetH * blockSize + offsetW) * inputChannels;
                for (int c = 0; c < inputChannels; c++) {
                    const int oc = c + offsetC;
                    const int offsetO = b * outputHeight * outputWidth * outputChannels 
                                        + oh * outputWidth * outputChannels + ow * outputChannels + oc;
                    const int offsetI = b * inputHeight * inputWidth * inputChannels
                                        + h * inputWidth * inputChannels + w * inputChannels + c;
                    outputDest[offsetO] = inputOrigin[offsetI];
                }
            }
        }
    }
    
    return NO_ERROR;
}

class SpaceToDepthCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        auto dataType   = inputs[0]->getType();
        if (dataType.bits == 32) {
            if (dataType.code == halide_type_int) {
                return new CPUSpaceToDepth<int32_t>(backend, op);
            }
            if (dataType.code == halide_type_float) {
                return new CPUSpaceToDepth<float>(backend, op);
            }
        }

        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(SpaceToDepthCreator, OpType_SpaceToDepth);

} // namespace MNN