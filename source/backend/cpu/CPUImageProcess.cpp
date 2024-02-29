//
//  CPUImageProcess.cpp
//  MNN
//
//  Created by MNN on 2021/10/27.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#include "backend/cpu/CPUImageProcess.hpp"
#include "compute/ImageProcessFunction.hpp"
#include <string.h>
#include <mutex>
#include "core/Macro.h"
#include <map>
#include <utility>

namespace MNN {
#define CACHE_SIZE 256

ErrorCode CPUImageProcess::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    int ih, iw, ic, oc, oh, ow;
    if (input->dimensions() == 3) {
        ih = input->length(0);
        iw = input->length(1);
        ic = input->length(2);
    } else {
        ih = input->height();
        iw = input->width();
        ic = input->channel();
    }
    mImgProc.reset(new ImageProcessUtils(mImgConfig));
    if (draw) {
        mImgProc->resizeFunc(ic, iw, ih, ic, iw, ih, inputs[0]->getType());
        return NO_ERROR;
    }
    auto output = outputs[0];
    oh = output->height();
    ow = output->width();
    oc = output->channel();
    mImgProc->resizeFunc(ic, iw, ih, oc, ow, oh, output->getType());

    return NO_ERROR;
}

ErrorCode CPUImageProcess::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto source = inputs[0]->host<uint8_t>();
    void* dest = nullptr;
    if (draw) {
        // change input to output
        dest = source;
    } else {
        dest = outputs[0]->host<void>();
    }
    mImgProc->execFunc(source, mStride, dest);
    return NO_ERROR;
}

class CPUImageProcessCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto process = op->main_as_ImageProcessParam();
        return new CPUImageProcess(backend, process);
    }
};

REGISTER_CPU_OP_CREATOR(CPUImageProcessCreator, OpType_ImageProcess);
} // namespace MNN
