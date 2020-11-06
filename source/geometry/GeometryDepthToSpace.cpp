//
//  GeometryDepthToSpace.cpp
//  MNN
//
//  Created by MNN on 2020/04/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
#include "core/Macro.h"
namespace MNN {
class GeometryDepthToSpace : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(inputs.size() == 1);
        const int blockSize   = op->main_as_DepthSpaceParam()->blockSize();
        auto mode = op->main_as_DepthSpaceParam()->mode();
        auto input            = inputs[0];
        auto output           = outputs[0];
        auto outputDes        = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        auto realTensor       = input;
        // For OpType_SpaceToDepth, swap input and output
        if (op->type() == OpType_SpaceToDepth) {
            auto temp = output;
            output    = input;
            input     = temp;
        }

        const int inHeight   = input->height();
        const int inWidth    = input->width();
        const int inChannel  = input->channel();
        const int outHeight  = output->height();
        const int outWidth   = output->width();
        const int outChannel = output->channel();
        // NCHW Stride
        int inputStride[4];
        int outputStride[4];
        if (MNN_DATA_FORMAT_NHWC == outputDes->dimensionFormat) {
            inputStride[0] = inWidth * inHeight * inChannel;
            inputStride[1] = 1;
            inputStride[2] = inWidth * inChannel;
            inputStride[3] = inChannel;

            outputStride[0] = outWidth * outHeight * outChannel;
            outputStride[1] = 1;
            outputStride[2] = outWidth * outChannel;
            outputStride[3] = outChannel;
        } else {
            inputStride[0] = inWidth * inHeight * inChannel;
            inputStride[1] = inWidth * inHeight;
            inputStride[2] = inWidth;
            inputStride[3] = 1;

            outputStride[0] = outWidth * outHeight * outChannel;
            outputStride[1] = outHeight * outWidth;
            outputStride[2] = outWidth;
            outputStride[3] = 1;
        }
        auto batch      = input->batch();
        auto regionSize = blockSize * blockSize * batch;
        outputDes->regions.resize(regionSize);
        for (int b = 0; b < batch; ++b) {
            auto dstB = b * outputStride[0];
            auto srcB = b * inputStride[0];
            for (int hb = 0; hb < blockSize; ++hb) {
                auto dstHB = dstB + hb * outputStride[2];
                for (int wb = 0; wb < blockSize; ++wb) {
                    auto dstWB        = dstHB + wb * outputStride[3];
                    int offsetC = hb * blockSize + wb;
                    if (mode == DepthToSpaceMode_DCR) {
                        offsetC *= outChannel;
                    }
                    auto srcWB        = srcB + offsetC * inputStride[1];

                    auto& region   = outputDes->regions[b * blockSize * blockSize + wb + hb * blockSize];
                    region.origin  = realTensor;
                    region.size[0] = inHeight;
                    region.size[1] = inWidth;
                    region.size[2] = outChannel;

                    auto srcR = &region.src;
                    auto dstR = &region.dst;
                    if (op->type() == OpType_SpaceToDepth) {
                        srcR = &region.dst;
                        dstR = &region.src;
                    }

                    dstR->offset    = dstWB;
                    dstR->stride[0] = outputStride[2] * blockSize;
                    dstR->stride[1] = outputStride[3] * blockSize;
                    dstR->stride[2] = outputStride[1];

                    srcR->offset    = srcWB;
                    srcR->stride[0] = inputStride[2];
                    srcR->stride[1] = inputStride[3];
                    srcR->stride[2] = inputStride[1];
                    if (mode == DepthToSpaceMode_CRD) {
                        srcR->stride[2] *= (blockSize * blockSize);
                    }
                }
            }
        }
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryDepthToSpace);
    GeometryComputer::registerGeometryComputer(comp, {OpType_DepthToSpace, OpType_SpaceToDepth});
}

REGISTER_GEOMETRY(GeometryDepthToSpace, _create);

} // namespace MNN
