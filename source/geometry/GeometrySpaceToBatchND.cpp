//
//  GeometrySpaceToBatchND.cpp
//  MNN
//
//  Created by MNN on 2020/04/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/Macro.h"
namespace MNN {
class GeometrySpaceToBatchND : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() == 1 || inputs.size() == 3);
        int blockSize = 0;
        const int *blockData, *paddingData;
        auto param            = op->main_as_SpaceBatch();
        if (inputs.size() == 3) {
            blockSize = inputs[1]->length(0);
            blockData = inputs[1]->host<int32_t>();
            paddingData = inputs[2]->host<int32_t>();
        } else {
            blockSize = param->blockShape()->dims()->data()[0];
            blockData = param->blockShape()->int32s()->data();
            paddingData = param->padding()->int32s()->data();
        }
        auto padTop           = paddingData[0];
        auto padLeft          = 0;
        auto blockShapeHeight = blockData[0];
        auto blockShapeWidth  = 1;
        if (blockSize > 1) {
            padLeft         = paddingData[2];
            blockShapeWidth = blockData[1];
        }
        auto input      = inputs[0];
        auto output     = outputs[0];
        auto outputDes  = TensorUtils::getDescribe(output);
        auto realTensor = input;
        // For OpType_BatchToSpaceND, swap input and output
        if (op->type() == OpType_BatchToSpaceND) {
            auto temp = output;
            output    = input;
            input     = temp;
        }

        const int inHeight  = input->height();
        const int inWidth   = input->width();
        const int inBatch   = input->batch();
        const int outHeight = output->height();
        const int outWidth  = output->width();
        const int outBatch  = output->batch();
        auto regionSize     = outBatch / inBatch;
        auto channel        = output->channel();
        outputDes->regions.resize(regionSize);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        // NCHW stride
        int inputStride[4];
        int outputStride[4];
        if (MNN_DATA_FORMAT_NHWC == outputDes->dimensionFormat) {
            inputStride[0] = inWidth * inHeight * channel;
            inputStride[1] = 1;
            inputStride[2] = inWidth * channel;
            inputStride[3] = channel;

            outputStride[0] = outWidth * outHeight * channel;
            outputStride[1] = 1;
            outputStride[2] = outWidth * channel;
            outputStride[3] = channel;
        } else {
            inputStride[0] = inWidth * inHeight * channel;
            inputStride[1] = inWidth * inHeight;
            inputStride[2] = inWidth;
            inputStride[3] = 1;

            outputStride[0] = outWidth * outHeight * channel;
            outputStride[1] = outHeight * outWidth;
            outputStride[2] = outWidth;
            outputStride[3] = 1;
        }
        for (int r = 0; r < regionSize; ++r) {
            auto& region  = outputDes->regions[r];
            region.origin = realTensor;
            int strideW   = r % blockShapeWidth;
            int strideH   = r / blockShapeWidth;

            const int validHStart = ALIMAX(0, (padTop - strideH + blockShapeHeight - 1) / blockShapeHeight);
            const int validHEnd =
                ALIMIN(outHeight, (inHeight + padTop - strideH + blockShapeHeight - 1) / blockShapeHeight);
            const int validWStart = ALIMAX(0, (padLeft - strideW + blockShapeWidth - 1) / blockShapeWidth);
            const int validWEnd =
                ALIMIN(outWidth, (inWidth + padLeft - strideW + blockShapeWidth - 1) / blockShapeWidth);
            int inHeightStart = validHStart * blockShapeHeight + strideH - padTop;
            int inWidthStart  = validHStart * blockShapeWidth + strideW - padLeft;
            auto srcR         = &region.src;
            auto dstR         = &region.dst;
            if (op->type() == OpType_BatchToSpaceND) {
                srcR = &region.dst;
                dstR = &region.src;
            }
            srcR->offset    = inHeightStart * inputStride[2] + inWidthStart * inputStride[3];
            srcR->stride[0] = 1 * inputStride[1];
            srcR->stride[1] = blockShapeHeight * inputStride[2];
            srcR->stride[2] = blockShapeWidth * inputStride[3];

            region.size[0] = inBatch * channel;
            region.size[1] = validHEnd - validHStart;
            region.size[2] = validWEnd - validWStart;

            dstR->offset =
                outputStride[2] * validHStart + outputStride[3] * validWStart + r * inBatch * outputStride[0];
            dstR->stride[0] = outputStride[1];
            dstR->stride[1] = outputStride[2];
            dstR->stride[2] = outputStride[3];
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometrySpaceToBatchND);
    GeometryComputer::registerGeometryComputer(comp, {OpType_SpaceToBatchND, OpType_BatchToSpaceND});
}

REGISTER_GEOMETRY(GeometrySpaceToBatchND, _create);

} // namespace MNN
