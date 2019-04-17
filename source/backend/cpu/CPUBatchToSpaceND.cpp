//
//  CPUBatchToSpaceND.cpp
//  MNN
//
//  Created by MNN on 2018/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBatchToSpaceND.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUBatchToSpaceND::CPUBatchToSpaceND(const Op* op, Backend* bn) : Execution(bn), mOp(op) {
    // nothing to do
}

ErrorCode CPUBatchToSpaceND::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mRun();
    return NO_ERROR;
}

ErrorCode CPUBatchToSpaceND::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    const int channelsDiv4   = UP_DIV(input->channel(), 4);
    const int inHeight       = input->height();
    const int inWidth        = input->width();
    const int inBatch        = input->batch();
    const int outHeight      = output->height();
    const int outWidth       = output->width();
    const int outBatch       = output->batch();
    const auto inputDataBase = input->host<float>();
    auto outDataBase         = output->host<float>();

    auto param                 = mOp->main_as_SpaceBatch();
    const int cropsTop         = param->padding()->int32s()->data()[0];
    const int cropsLeft        = param->padding()->int32s()->data()[2];
    const int blockShapeHeight = param->blockShape()->int32s()->data()[0];
    const int blockShapeWidth  = param->blockShape()->int32s()->data()[1];
    mRun                       = [=]() {
        for (int ib = 0; ib < inBatch; ++ib) {
            const int ob            = ib % outBatch;
            const int spatialOffset = ib / outBatch;
            const int strideH       = spatialOffset / blockShapeWidth;
            const int strideW       = spatialOffset % blockShapeWidth;

            auto inDataBatch  = inputDataBase + ib * channelsDiv4 * inHeight * inWidth * 4;
            auto outDataBatch = outDataBase + ob * channelsDiv4 * outHeight * outWidth * 4;

            const int validHStart = ALIMAX(0, (cropsTop - strideH + blockShapeHeight - 1) / blockShapeHeight);
            const int validHEnd =
                ALIMIN(inHeight, (outHeight + cropsTop - strideH + blockShapeHeight - 1) / blockShapeHeight);
            const int validWStart = ALIMAX(0, (cropsLeft - strideW + blockShapeWidth - 1) / blockShapeWidth);
            const int validWEnd =
                ALIMIN(inWidth, (outWidth + cropsLeft - strideW + blockShapeWidth - 1) / blockShapeWidth);

            for (int c = 0; c < channelsDiv4; ++c) {
                auto inDataChannel  = inDataBatch + c * inHeight * inWidth * 4;
                auto outDataChannel = outDataBatch + c * outHeight * outWidth * 4;

                for (int h = validHStart; h < validHEnd; ++h) {
                    const int heightIndex = h * blockShapeHeight + strideH - cropsTop;
                    const int widthIndex  = validWStart * blockShapeWidth + strideW - cropsLeft;
                    auto inDataHeight     = inDataChannel + h * inWidth * 4;
                    auto outDataHeight    = outDataChannel + (heightIndex * outWidth + widthIndex) * 4;

                    MNNCopyC4WithStride(inDataHeight + validWStart * 4, outDataHeight, 4, blockShapeWidth * 4,
                                        validWEnd - validWStart);
                }
            }
        }
    };

    return NO_ERROR;
}

class CPUBatchToSpaceNDCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUBatchToSpaceND(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUBatchToSpaceNDCreator, OpType_BatchToSpaceND);
} // namespace MNN
