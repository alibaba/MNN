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

static void _NC4HW42NHWCUint8(const uint8_t* dest, uint8_t* source, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNC4HW4ToNHWCUint8(srcBatch, dstBatch, area, c);
    }
}

static void _NHWC2NC4HW4Uint8(const uint8_t* source, uint8_t* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNHWCToNC4HW4Uint8(dstBatch, srcBatch, area, c);
    }
}

void CPUTensorConverter::NC4HW42NHWC(const float* dest, float* source, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNC4HW4ToNHWC(srcBatch, dstBatch, area, c);
    }
}

void CPUTensorConverter::NHWC2NC4HW4(const float* source, float* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = ALIGN_UP4(c) * area;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        MNNTensorConvertNHWCToNC4HW4(dstBatch, srcBatch, area, c);
    }
}

void CPUTensorConverter::NCHW2NHWC(const float* source, float* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i;
            auto dstArea = dstBatch + i * c;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci] = srcArea[ci * area];
            }
        }
    }
}

void CPUTensorConverter::NHWC2NCHW(const float* source, float* dest, int b, int c, int area) {
    int sourceBatchsize = c * area;
    int destBatchSize   = sourceBatchsize;
    for (int bi = 0; bi < b; ++bi) {
        auto srcBatch = source + bi * sourceBatchsize;
        auto dstBatch = dest + bi * destBatchSize;
        for (int i = 0; i < area; ++i) {
            auto srcArea = srcBatch + i * c;
            auto dstArea = dstBatch + i;
            for (int ci = 0; ci < c; ++ci) {
                dstArea[ci * area] = srcArea[ci];
            }
        }
    }
}

ErrorCode CPUTensorConverter::convert(const Tensor* input, const Tensor* output) {
    auto ib     = input->buffer();
    auto ob     = output->buffer();
    auto source = TensorUtils::getDescribe(input)->dimensionFormat;
    auto dest   = TensorUtils::getDescribe(output)->dimensionFormat;
    if (ib.dimensions <= 1 || source == dest) {
        ::memcpy(ob.host, ib.host, input->size());
        return NO_ERROR;
    }
    if (source == MNN_DATA_FORMAT_UNKNOWN || dest == MNN_DATA_FORMAT_UNKNOWN) {
        MNN_ERROR("unknown data format!\nsrc: %d, dst: %d\n", source, dest);
        return INVALID_VALUE;
    }
    int area = 1, batch = ib.dim[0].extent, channel;
    if (source == MNN_DATA_FORMAT_NC4HW4 || source == MNN_DATA_FORMAT_NCHW) {
        channel = ib.dim[1].extent;
        for (int axis = 2; axis < ib.dimensions; ++axis) {
            area *= ib.dim[axis].extent;
        }
    } else {
        channel = ib.dim[ib.dimensions - 1].extent;
        for (int axis = 1; axis < ib.dimensions - 1; ++axis) {
            area *= ib.dim[axis].extent;
        }
    }
    const int bitLength = ib.type.bytes();
    
    if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NCHW == dest) {
        if (bitLength == 1) {
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNUnpackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i,
                                 (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }
        MNN_ASSERT(bitLength == 4);
        for (int i = 0; i < ib.dim[0].extent; ++i) {
            MNNUnpackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
        }
        return NO_ERROR;
    }

    if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        if (bitLength == 1) {
            for (int i = 0; i < ib.dim[0].extent; ++i) {
                MNNPackC4Uint8((uint8_t*)ob.host + ob.dim[0].stride * i, (const uint8_t*)ib.host + ib.dim[0].stride * i, area, channel);
            }
            return NO_ERROR;
        }
        MNN_ASSERT(bitLength == 4);
        for (int i = 0; i < ib.dim[0].extent; ++i) {
            MNNPackC4((float*)ob.host + ob.dim[0].stride * i, (const float*)ib.host + ib.dim[0].stride * i, area, channel);
        }
        return NO_ERROR;
    }
    
    if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NC4HW4 == dest) {
        if (bitLength == 1) {
            _NHWC2NC4HW4Uint8((uint8_t*)ib.host, (uint8_t*)ob.host, batch, channel, area);
        } else {
            NHWC2NC4HW4((float*)ib.host, (float*)ob.host, batch, channel, area);
        }
    } else if (MNN_DATA_FORMAT_NC4HW4 == source && MNN_DATA_FORMAT_NHWC == dest) {
        if (bitLength == 1) {
            _NC4HW42NHWCUint8((uint8_t*)ib.host, (uint8_t*)ob.host, batch, channel, area);
        } else {
            NC4HW42NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
        }
    } else if (MNN_DATA_FORMAT_NHWC == source && MNN_DATA_FORMAT_NCHW == dest) {
        MNN_ASSERT(bitLength == 4);
        NHWC2NCHW((float*)ib.host, (float*)ob.host, batch, channel, area);
    } else if (MNN_DATA_FORMAT_NCHW == source && MNN_DATA_FORMAT_NHWC == dest) {
        MNN_ASSERT(bitLength == 4);
        NCHW2NHWC((float*)ib.host, (float*)ob.host, batch, channel, area);
    } else {
        MNN_ASSERT(false);
        return NOT_SUPPORT;
    }
    
    return NO_ERROR;
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
