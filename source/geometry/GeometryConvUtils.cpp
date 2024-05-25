//
//  GeometryConvUtils.cpp
//  MNN
//
//  Created by MNN on 2020/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GeometryConvUtils.hpp"
#include "ConvertUtils.hpp"

#define ADD_PAD_VALUE(POS, OFFSET, NUM, STRIDE)               \
    if (POS##Pad > 0) {                                       \
        Tensor::InsideDescribe::Region region;                \
        region.origin        = padVal;                        \
        region.size[0]       = ic;                            \
        region.size[1]       = NUM;                           \
        region.size[2]       = POS##Pad;                      \
        region.src.offset    = 0;                             \
        region.dst.offset    = dstOffsetKx + (OFFSET);        \
        region.src.stride[0] = 0;                             \
        region.src.stride[1] = 0;                             \
        region.src.stride[2] = 0;                             \
        region.dst.stride[0] = dstStrideChannel;              \
        region.dst.stride[1] = STRIDE;                        \
        region.dst.stride[2] = 1;                             \
        des->regions.emplace_back(std::move(region));         \
    }
namespace MNN {
flatbuffers::Offset<Op> GeometryConvUtils::makeRelu6(flatbuffers::FlatBufferBuilder& builder, float minValue, float maxValue) {
    Relu6Builder relu6B(builder);
    relu6B.add_maxValue(maxValue);
    relu6B.add_minValue(minValue);
    auto paOffset = relu6B.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_ReLU6);
    opB.add_main_type(OpParameter_Relu6);
    opB.add_main(paOffset);
    return opB.Finish();
}
void GeometryConvUtils::im2Col3d(Tensor* im2Col, Tensor* input, int ic, int kd, int kh, int kw, int batch, int od, int oh, int ow,
    int id, int ih, int iw, int sd, int sh, int sw, int dd, int dh, int dw, int pd, int ph, int pw, int srcKernelOffset) {
    im2Col->buffer().type       = halide_type_of<float>();
    im2Col->buffer().dimensions = 2;
    im2Col->setLength(0, ic * kd * kh * kw);
    im2Col->setLength(1, batch * od * oh * ow);
    TensorUtils::setLinearLayout(im2Col);
    auto des             = TensorUtils::getDescribe(im2Col);
    des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    des->regions.clear();
    des->regions.reserve(batch * ic * kd * kh * kw);
    for (int c = 0; c < ic; ++c) {
        for (int n = 0; n < batch; ++n) {
            auto dstOffset = (c * kd * kh * kw * batch + n) * od * oh * ow;
            // auto dstOffset = (n * ic + c) * od * oh * ow * kd * kh * kw;
            auto srcOffset = (n * ic + c) * id * ih * iw;
            for (int kz = 0; kz < kd; ++kz) {
                auto startSz = kz * dd - pd;
                int startDz = 0;
                if (startSz < 0) {
                    startDz = ((-startSz) + sd - 1) / sd;
                    startSz = startSz + startDz * sd;
                }
                auto endDz = od - 1;
                auto endSz = endDz * sd + kz * dd - pd;
                if (endSz >= id) {
                    endDz = endDz - (endSz - id + sd) / sd;
                    endSz = endDz * sd + kz * dd - pd;
                }
                if (startDz > endDz || endDz < 0 || startSz >= id) {
                    continue;
                }
                auto dstOffsetKz = dstOffset + kz * kw * kh * ow * oh * od * batch + startDz * oh *  ow;
                auto srcOffsetKz = srcOffset + startSz * ih * iw;
                for (int ky = 0; ky < kh; ++ky) {
                    auto startSy = ky * dh - ph;
                    int startDy  = 0;
                    if (startSy < 0) {
                        startDy = ((-startSy) + sh - 1) / sh;
                        startSy = startSy + startDy * sh;
                    }
                    auto endDy = oh - 1;
                    auto endSy = endDy * sh + ky * dh - ph;
                    if (endSy >= ih) {
                        endDy = endDy - (endSy - ih + sh) / sh;
                        endSy = endDy * sh + ky * dh - ph;
                    }
                    if (startDy > endDy || endDy < 0 || startSy >= ih) {
                        continue;
                    }
                    auto dstOffsetKy = dstOffsetKz + ky * kw * ow * oh * od * batch + startDy * ow;
                    auto srcOffsetKy = srcOffsetKz + startSy * iw;
                    for (int kx = 0; kx < kw; ++kx) {
                        auto startSx = kx * dw - pw;
                        int startDx  = 0;
                        if (startSx < 0) {
                            startDx = ((-startSx) + sw - 1) / sw;
                            startSx = startSx + startDx * sw;
                        }
                        auto endDx = ow - 1;
                        auto endSx = endDx * sw + kx * dw - pw;
                        if (endSx >= iw) {
                            endDx = endDx - (endSx - iw + sw) / sw;
                            endSx = endDx * sw + kx * dw - pw;
                        }
                        if (startDx > endDx || endDx < 0 || startSx >= iw) {
                            continue;
                        }
                        auto dstOffsetKx = dstOffsetKy + kx * od * oh * ow * batch + startDx;
                        auto srcOffsetKx = srcOffsetKy + startSx + srcKernelOffset * (kx + ky * kw);
                        Tensor::InsideDescribe::Region region;
                        region.origin        = input;
                        region.size[0]       = endDz - startDz + 1;
                        region.size[1]       = endDy - startDy + 1;
                        region.size[2]       = endDx - startDx + 1;
                        region.src.offset    = srcOffsetKx;
                        region.dst.offset    = dstOffsetKx;
                        region.src.stride[0] = sd * ih * iw;
                        region.dst.stride[0] = oh * ow;
                        region.src.stride[1] = sh * iw;
                        region.dst.stride[1] = ow;
                        region.src.stride[2] = sw;
                        region.dst.stride[2] = 1;
                        des->regions.emplace_back(std::move(region));
                    }
                }
                // MNN_ASSERT(des->regions.size() > 0);
            }
        }
    }
}
std::shared_ptr<Tensor> GeometryConvUtils::im2Col(Tensor* im2Col, Tensor* input, int ic, int kh, int kw, int batch, int oh, int ow, int ih,
                               int iw, int sh, int sw, int dh, int dw, std::pair<int, int> pads, int srcKernelOffset, Tensor* padVal) {
    im2Col->buffer().type       = halide_type_of<float>();
    im2Col->buffer().dimensions = 2;
    im2Col->setLength(0, ic * kw * kh);
    im2Col->setLength(1, batch * ow * oh);
    TensorUtils::setLinearLayout(im2Col);
    auto des             = TensorUtils::getDescribe(im2Col);
    des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
    des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    des->regions.clear();
    std::shared_ptr<Tensor> tempTensor;
    if (batch > 1) {
        tempTensor.reset(new Tensor);
        tempTensor->buffer().type       = halide_type_of<float>();
        tempTensor->buffer().dimensions = 2;
        tempTensor->setLength(0, kw * kh * ic * batch);
        tempTensor->setLength(1, ow * oh);
        TensorUtils::setLinearLayout(tempTensor.get());
        des = TensorUtils::getDescribe(tempTensor.get());
        des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
    }
    des->regions.reserve(kw * kh);
    int dstStrideChannel = oh * ow * kh * kw;
    int srcStrideChannel = iw * ih;
    auto dstOffset = 0;
    auto srcOffset = 0;
    for (int ky = 0; ky < kh; ++ky) {
        auto startSy = ky * dh - pads.second;
        int startDy  = 0;
        int upPad = 0, belowPad = 0;
        if (startSy < 0) {
            startDy = ((-startSy) + sh - 1) / sh;
            startSy = startSy + startDy * sh;
            upPad = startDy * ow;
        }
        auto endDy = oh - 1;
        auto endSy = endDy * sh + ky * dh - pads.second;
        if (endSy >= ih) {
            endDy = endDy - (endSy - ih + sh) / sh;
            endSy = endDy * sh + ky * dh - pads.second;
            belowPad = (oh - endDy - 1) * ow;
        }
        if (startDy > endDy || endDy < 0 || startSy >= ih) {
            continue;
        }
        auto dstOffsetKy = dstOffset + ky * kw * ow * oh + startDy * ow;
        auto srcOffsetKy = srcOffset + startSy * iw;
        for (int kx = 0; kx < kw; ++kx) {
            auto startSx = kx * dw - pads.first;
            int startDx  = 0;
            int leftPad = 0, rightPad = 0;
            if (startSx < 0) {
                startDx = ((-startSx) + sw - 1) / sw;
                startSx = startSx + startDx * sw;
                leftPad = startDx;
            }
            auto endDx = ow - 1;
            auto endSx = endDx * sw + kx * dw - pads.first;
            if (endSx >= iw) {
                endDx = endDx - (endSx - iw + sw) / sw;
                endSx = endDx * sw + kx * dw - pads.first;
                rightPad = ow - endDx - 1;
            }
            if (startDx > endDx || endDx < 0 || startSx >= iw) {
                continue;
            }
            auto dstOffsetKx = dstOffsetKy + kx * ow * oh + startDx;
            auto srcOffsetKx = srcOffsetKy + startSx + srcKernelOffset * (kx + ky * kw);
            const int ohExcludePad = endDy - startDy + 1;
            const int owExcludePad = endDx - startDx + 1;
            // if given padVal, pad value will use padVa otherwise use zero
            if (padVal) {
                ADD_PAD_VALUE(up, -(startDx+upPad), 1, 0);
                ADD_PAD_VALUE(below, ohExcludePad * ow - startDx, 1, 0);
                ADD_PAD_VALUE(left, -leftPad, ohExcludePad, ow);
                ADD_PAD_VALUE(right, owExcludePad, ohExcludePad, ow);
            }
            Tensor::InsideDescribe::Region region;
            region.origin        = input;
            region.size[0]       = ic * batch;
            region.size[1]       = ohExcludePad;
            region.size[2]       = owExcludePad;
            region.src.offset    = srcOffsetKx;
            region.dst.offset    = dstOffsetKx;
            region.src.stride[0] = srcStrideChannel;
            region.dst.stride[0] = dstStrideChannel;
            region.src.stride[1] = sh * iw;
            region.dst.stride[1] = ow;
            region.src.stride[2] = sw;
            region.dst.stride[2] = 1;
            des->regions.emplace_back(std::move(region));
        }
    }
    if (batch > 1) {
        // Transpose: batch, ic, kh, kw, oh, ow -> ic, kh, kw, batch, oh, ow
        auto destDes = TensorUtils::getDescribe(im2Col);
        destDes->regions.resize(1);
        auto& reg = destDes->regions[0];
        reg.size[0] = ic * kh * kw;
        reg.size[1] = batch;
        reg.size[2] = oh * ow;
        reg.src.offset = 0;
        reg.src.stride[0] = reg.size[2];
        reg.src.stride[1] = reg.size[2] * reg.size[0];
        reg.src.stride[2] = 1;

        reg.dst.offset = 0;
        reg.dst.stride[0] = reg.size[2] * batch;
        reg.dst.stride[1] = reg.size[2];
        reg.dst.stride[2] = 1;
        reg.origin = tempTensor.get();
    }
    return tempTensor;
}
bool GeometryConvUtils::computeSingle(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, GeometryComputer::Context& context, CommandBuffer& res) {
    auto newOutputs   = outputs;
    auto newInputs    = inputs;
    auto originOutput = outputs[0];
    auto output       = originOutput;
    auto inputDes     = TensorUtils::getDescribe(newInputs[0]);
    auto format       = inputDes->dimensionFormat;
    if (MNN_DATA_FORMAT_NC4HW4 != format) {
        std::shared_ptr<Tensor> newInput(new Tensor(newInputs[0], Tensor::CAFFE_C4, false));
        ConvertUtils::compute(newInputs[0], newInput.get(), res);
        newInputs[0] = newInput.get();
        res.extras.emplace_back(std::move(newInput));
        std::shared_ptr<Tensor> newOutput(new Tensor(originOutput, Tensor::CAFFE_C4, false));
        output        = newOutput.get();
        newOutputs[0] = output;
        res.extras.emplace_back(newOutput);
    }
    std::shared_ptr<Command> cmd(new Command);
    cmd->op      = op;
    cmd->inputs  = std::move(newInputs);
    cmd->outputs = std::move(newOutputs);
    res.command.emplace_back(std::move(cmd));
    if (originOutput != output) {
        ConvertUtils::compute(output, originOutput, res);
    }
    return true;
}
}; // namespace MNN
