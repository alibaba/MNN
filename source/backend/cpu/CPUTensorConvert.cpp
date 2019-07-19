//
//  CPUTensorConvert.cpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUTensorConvert.hpp"
#include "CPUBackend.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {

static void _NC4HW42NHWCUint8(const uint8_t* dest, uint8_t* source, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = ALIGN_UP4(c) * w * h;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNC4HW4ToNHWCUint8(srcBatch, dstBatch, w * h, c);
    }
}

static void _NHWC2NC4HW4Uint8(const uint8_t* source, uint8_t* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = ALIGN_UP4(c) * w * h;
    int area            = w * h;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNHWCToNC4HW4Uint8(dstBatch, srcBatch, area, c);
    }
}

void CPUTensorConverter::NC4HW42NHWC(const float* dest, float* source, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = ALIGN_UP4(c) * w * h;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNC4HW4ToNHWC(srcBatch, dstBatch, w * h, c);
    }
}

void CPUTensorConverter::NHWC2NC4HW4(const float* source, float* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = ALIGN_UP4(c) * w * h;
    int area            = w * h;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNHWCToNC4HW4(dstBatch, srcBatch, area, c);
    }
}

void CPUTensorConverter::NCHW2NHWC(const float* source, float* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int hi = 0; hi < h; ++hi) {
            auto srcHeight = srcBatch + hi * w;
            auto dstHeight = dstBatch + hi * w * c;
            for (int wi = 0; wi < w; ++wi) {
                auto srcWidth = srcHeight + wi;
                auto dstWidth = dstHeight + wi * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstWidth[ci] = srcWidth[ci * w * h];
                }
            }
        }
    }
}

void CPUTensorConverter::NHWC2NCHW(const float* source, float* dest, int b, int h, int w, int c) {
    int sourceBatchsize = h * w * c;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int hi = 0; hi < h; ++hi) {
            auto srcHeight = srcBatch + hi * w * c;
            auto dstHeight = dstBatch + hi * w;
            for (int wi = 0; wi < w; ++wi) {
                auto dstWidth = dstHeight + wi;
                auto srcWidth = srcHeight + wi * c;
                for (int ci = 0; ci < c; ++ci) {
                    dstWidth[ci * w * h] = srcWidth[ci];
                }
            }
        }
    }
}

ErrorCode CPUTensorConverter::convert(const Tensor* input, const Tensor* output) {
    auto ib     = input->buffer();
    auto ob     = output->buffer();
    auto source = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
    if (ib.dimensions < 1 || source == dest) {
        ::memcpy(ob.host, ib.host, input->size());
        return NO_ERROR;
    }
    int area = 1;
    for (int axis = 2; axis < ib.dimensions; ++axis) {
        area *= ib.dim[axis].extent;
    }
    if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
        for (int i = 0; i < ib.dim[0].extent; ++i) {
            MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area,
                        ib.dim[1].extent);
        }
        return NO_ERROR;
    }

    if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        for (int i = 0; i < ib.dim[0].extent; ++i) {
            MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area,
                      ib.dim[1].extent);
        }
        return NO_ERROR;
    }
    if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        int b = ib.dim[0].extent;
        int h = ib.dim[1].extent;
        int w = ib.dim[2].extent;
        int c = ib.dim[3].extent;
        if (ib.type.bytes() == 1) {
            _NHWC2NC4HW4Uint8((uint8_t*)ib.host, (uint8_t*)ob.host, b, h, w, c);
            return NO_ERROR;
        }
        NHWC2NC4HW4((float*)ib.host, (float*)ob.host, b, h, w, c);
        return NO_ERROR;
    } else if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NHWC == dest) {
        int b = ob.dim[0].extent;
        int h = ob.dim[1].extent;
        int w = ob.dim[2].extent;
        int c = ob.dim[3].extent;
        if (ib.type.bytes() == 1) {
            _NC4HW42NHWCUint8((uint8_t*)ib.host, (uint8_t*)ob.host, b, h, w, c);
            return NO_ERROR;
        }
        NC4HW42NHWC((float*)ib.host, (float*)ob.host, b, h, w, c);
        return NO_ERROR;
    } else if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
        int b = ib.dim[0].extent;
        int h = ib.dim[1].extent;
        int w = ib.dim[2].extent;
        int c = ib.dim[3].extent;
        NHWC2NCHW((float*)ib.host, (float*)ob.host, b, h, w, c);
        return NO_ERROR;
    } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
        int b = ob.dim[0].extent;
        int h = ob.dim[1].extent;
        int w = ob.dim[2].extent;
        int c = ob.dim[3].extent;
        NCHW2NHWC((float*)ib.host, (float*)ob.host, b, h, w, c);
        return NO_ERROR;
    }

    MNN_ASSERT(false);

    return NOT_SUPPORT;
}

ErrorCode CPUTensorConverter::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return convert(inputs[0], outputs[0]);
}

class CPUTensorConvertFactory : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        return new CPUTensorConverter(backend, op->main_as_TensorConvertInfo()->source(),
                                      op->main_as_TensorConvertInfo()->dest());
    }
};

REGISTER_CPU_OP_CREATOR(CPUTensorConvertFactory, OpType_ConvertTensor);
} // namespace MNN
