//
//  CPUDepthToSpace.hpp
//  MNN
//
//  Created by MNN on 2019/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDepthToSpace.hpp"
#include "Backend.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"

namespace MNN {

template <typename T>
CPUDepthToSpace<T>::CPUDepthToSpace(Backend* backend, const MNN::Op* op) : Execution(backend), mOp(op) {
    // do nothing
}

template <typename T>
ErrorCode CPUDepthToSpace<T>::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    // do nothing

    return NO_ERROR;
}

template <typename T>
ErrorCode CPUDepthToSpace<T>::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    const int blockSize = mOp->main_as_DepthSpaceParam()->blockSize();

    const int inputHeight = ib.dim[1].extent;
    const int inputWidth = ib.dim[2].extent;
    const int inputChannels = ib.dim[3].extent;

    const int outputBatch = ob.dim[0].extent;
    const int outputHeight = ob.dim[1].extent;
    const int outputWidth = ob.dim[2].extent;
    const int outputChannels = ob.dim[3].extent;

    T* inputOrigin = reinterpret_cast<T*>(ib.host);
    T* outputDest = reinterpret_cast<T*>(ob.host);

    // NHWC
    // TODO: implement NC4HW4
    for (int b = 0; b < outputBatch; b++) {
        for (int h = 0; h < outputHeight; h++) {
            const int ih = h / blockSize;
            const int offsetH = h % blockSize;
            for (int w = 0; w < outputWidth; w++) {
                const int iw = w / blockSize;
                const int offsetW = w % blockSize;
                const int offsetC = (offsetH * blockSize + offsetW) * outputChannels;
                for (int c = 0; c < outputChannels; c++) {
                    const int ic = c + offsetC;
                    const int offsetO = b * outputHeight * outputWidth * outputChannels 
                                        + h * outputWidth * outputChannels + w * outputChannels + c;
                    const int offsetI = b * inputHeight * inputWidth * inputChannels
                                        + ih * inputWidth * inputChannels + iw * inputChannels + ic;
                    outputDest[offsetO] = inputOrigin[offsetI];
                }
            }
        }
    }
    
    return NO_ERROR;
}

class DepthToSpaceCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        auto dataType   = inputs[0]->getType();
        if (dataType.bits == 32) {
            if (dataType.code == halide_type_int) {
                return new CPUDepthToSpace<int32_t>(backend, op);
            }
            if (dataType.code == halide_type_float) {
                return new CPUDepthToSpace<float>(backend, op);
            }
        }

        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(DepthToSpaceCreator, OpType_DepthToSpace);

} // namespace MNN
