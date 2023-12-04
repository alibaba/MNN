//
//  GeometryConv2DBackPropFilter.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "GeometryConvUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
class GeometryConv2DBackPropFilter : public GeometryComputer {
public:
    bool computeForDepthWise(const Convolution2DCommon* common, Tensor* input, Tensor* outputDiff, Tensor* kernelDiff,
                             Context& context, CommandBuffer& res) const {
        auto kw    = common->kernelX();
        auto kh    = common->kernelY();
        auto sw    = common->strideX();
        auto sh    = common->strideY();
        auto dw    = common->dilateX();
        auto dh    = common->dilateY();
        auto batch = outputDiff->batch();
        auto ow    = outputDiff->width();
        auto oh    = outputDiff->height();
        auto ic    = input->channel();
        auto iw    = input->width();
        auto ih    = input->height();
        auto pads  = ConvolutionCommon::convolutionPad(input, outputDiff, common);
        if (TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NCHW) {
            std::shared_ptr<Tensor> newT(new Tensor(input, Tensor::CAFFE, false));
            ConvertUtils::compute(input, newT.get(), res);
            input = newT.get();
            res.extras.emplace_back(newT);
        }
        if (TensorUtils::getDescribe(outputDiff)->dimensionFormat != MNN_DATA_FORMAT_NCHW) {
            std::shared_ptr<Tensor> newT(new Tensor(outputDiff, Tensor::CAFFE, false));
            ConvertUtils::compute(outputDiff, newT.get(), res);
            outputDiff = newT.get();
            res.extras.emplace_back(newT);
        }
        auto outputDes        = TensorUtils::getDescribe(kernelDiff);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        outputDes->regions.clear();
        for (int ky = 0; ky < kh; ++ky) {
            auto startSy = ky * dh - pads.second;
            int startDy  = 0;
            if (startSy < 0) {
                startDy = ((-startSy) + sh - 1) / sh;
                startSy = startSy + startDy * sh;
            }
            auto endDy = oh - 1;
            auto endSy = endDy * sh + ky * dh - pads.second;
            if (endSy >= ih) {
                endDy = endDy - (endSy - ih + sh) / sh;
                endSy = endDy * sh + ky * dh - pads.second;
            }
            if (startDy > endDy) {
                continue;
            }
            MNN_ASSERT(endDy >= 0);
            MNN_ASSERT(startDy < ih);
            auto dstOffsetKy = startDy * ow;
            auto srcOffsetKy = startSy * iw;
            for (int kx = 0; kx < kw; ++kx) {
                auto startSx = kx * dw - pads.first;
                int startDx  = 0;
                if (startSx < 0) {
                    startDx = ((-startSx) + sw - 1) / sw;
                    startSx = startSx + startDx * sw;
                }
                auto endDx = ow - 1;
                auto endSx = endDx * sw + kx * dw - pads.first;
                if (endSx >= iw) {
                    endDx = endDx - (endSx - iw + sw) / sw;
                    endSx = endDx * sw + kx * dw - pads.first;
                }
                if (startDy > endDy) {
                    continue;
                }
                auto dstOffsetKx = dstOffsetKy + startDx;
                auto srcOffsetKx = srcOffsetKy + startSx;
                // Sampler
                std::shared_ptr<Tensor> inputTensor(new Tensor(outputDiff, Tensor::CAFFE, false));
                auto des        = TensorUtils::getDescribe(inputTensor.get());
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->regions.resize(1);
                {
                    Tensor::InsideDescribe::Region& region = des->regions[0];
                    region.origin                          = input;
                    region.size[0]                         = batch * ic;
                    region.size[1]                         = endDy - startDy + 1;
                    region.size[2]                         = endDx - startDx + 1;
                    region.src.offset                      = srcOffsetKx;
                    region.dst.offset                      = dstOffsetKx;
                    region.src.stride[0]                   = iw * ih;
                    region.dst.stride[0]                   = ow * oh;
                    region.src.stride[1]                   = sh * iw;
                    region.dst.stride[1]                   = ow;
                    region.src.stride[2]                   = sw;
                    region.dst.stride[2]                   = 1;
                    res.extras.emplace_back(inputTensor);
                }

                auto currentTensor = inputTensor.get();
                // Multi
                {
                    std::shared_ptr<Tensor> newTensor(new Tensor(outputDiff, Tensor::CAFFE, false));
                    auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, currentTensor, outputDiff,
                                                                 newTensor.get());
                    res.command.emplace_back(std::move(cmd));
                    res.extras.emplace_back(newTensor);
                    currentTensor = newTensor.get();
                }
                // Reduce - 0
                {
                    std::shared_ptr<Tensor> reduceInputTensor(
                        Tensor::createDevice<float>({batch * ic, ow * oh, 1}, Tensor::CAFFE));
                    {
                        auto inputDes        = TensorUtils::getDescribe(reduceInputTensor.get());
                        inputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                        inputDes->regions    = {TensorUtils::makeFullSlice(currentTensor)};
                    }
                    std::shared_ptr<Tensor> reduceOutputTensor(
                        Tensor::createDevice<float>({batch * ic, 1, 1}, Tensor::CAFFE));
                    auto cmd      = GeometryComputerUtils::makeReduce(ReductionType_SUM, reduceInputTensor.get(),
                                                                 reduceOutputTensor.get());
                    currentTensor = reduceOutputTensor.get();
                    res.command.emplace_back(std::move(cmd));
                    res.extras.emplace_back(reduceInputTensor);
                    res.extras.emplace_back(reduceOutputTensor);
                }
                // Reduce - 1
                {
                    std::shared_ptr<Tensor> reduceInputTensor(
                        Tensor::createDevice<float>({1, batch, ic}, Tensor::CAFFE));
                    {
                        auto inputDes        = TensorUtils::getDescribe(reduceInputTensor.get());
                        inputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                        inputDes->regions    = {TensorUtils::makeFullSlice(currentTensor)};
                    }
                    std::shared_ptr<Tensor> reduceOutputTensor(Tensor::createDevice<float>({1, 1, ic}, Tensor::CAFFE));
                    currentTensor = reduceOutputTensor.get();
                    auto cmd      = GeometryComputerUtils::makeReduce(ReductionType_SUM, reduceInputTensor.get(),
                                                                 reduceOutputTensor.get());
                    res.command.emplace_back(std::move(cmd));
                    res.extras.emplace_back(reduceInputTensor);
                    res.extras.emplace_back(reduceOutputTensor);
                }
                // Set to output
                Tensor::InsideDescribe::Region region;
                region.origin        = currentTensor;
                region.size[0]       = 1;
                region.size[1]       = 1;
                region.size[2]       = ic;
                region.dst.offset    = ky * kw + kx;
                region.dst.stride[0] = 0;
                region.dst.stride[1] = 0;
                region.dst.stride[2] = kh * kw;
                region.src.offset    = 0;
                region.src.stride[0] = 0;
                region.src.stride[1] = 0;
                region.src.stride[2] = 1;
                outputDes->regions.emplace_back(std::move(region));
            }
        }
        return true;
    }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto common     = op->main_as_Convolution2D()->common();
        auto input      = inputs[0];
        auto outputDiff = inputs[1];
        bool depthWise  = false;
        if (inputs[0]->channel() == inputs[1]->channel() && inputs[1]->channel() == common->group()) {
            depthWise = true;
            return computeForDepthWise(common, input, outputDiff, outputs[0], context, res);
        }
        auto kw    = common->kernelX();
        auto kh    = common->kernelY();
        auto sw    = common->strideX();
        auto sh    = common->strideY();
        auto dw    = common->dilateX();
        auto dh    = common->dilateY();
        auto batch = outputDiff->batch();
        auto ow    = outputDiff->width();
        auto oh    = outputDiff->height();
        auto oc    = outputDiff->channel();
        auto ic    = input->channel();
        auto iw    = input->width();
        auto ih    = input->height();
        auto pads  = ConvolutionCommon::convolutionPad(input, outputDiff, common);
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        MNN_ASSERT(TensorUtils::getDescribe(outputDiff)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        Tensor* A = nullptr;
        Tensor* B = nullptr;
        {
            // B: Input Im2Col, n, ic, ih, iw -> ic*kh*kw, n*oh*ow
            std::shared_ptr<Tensor> im2Col(new Tensor);
            std::shared_ptr<Tensor> im2ColTemp = GeometryConvUtils::im2Col(im2Col.get(), input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
            if (im2ColTemp.get() != nullptr) {
                res.extras.emplace_back(im2ColTemp);
            }
            B = im2Col.get();
            res.extras.emplace_back(im2Col);
        }
        {
            // A: Output n, oc, oh, ow -> oc, n*oh*ow
            std::shared_ptr<Tensor> outputTranspose(new Tensor);
            A                                    = outputTranspose.get();
            outputTranspose->buffer().type       = halide_type_of<float>();
            outputTranspose->buffer().dimensions = 2;
            outputTranspose->setLength(0, oc);
            outputTranspose->setLength(1, batch * ow * oh);
            auto des = TensorUtils::getDescribe(outputTranspose.get());
            des->regions.resize(1);
            des->memoryType   = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto& reg         = des->regions[0];
            reg.origin        = outputDiff;
            reg.size[0]       = oc;
            reg.size[1]       = batch;
            reg.size[2]       = ow * oh;
            reg.src.offset    = 0;
            reg.src.stride[0] = oh * ow;
            reg.src.stride[1] = oh * ow * oc;
            reg.src.stride[2] = 1;
            reg.dst.offset    = 0;
            reg.dst.stride[0] = oh * ow * batch;
            reg.dst.stride[1] = oh * ow;
            reg.dst.stride[2] = 1;
            res.extras.emplace_back(std::move(outputTranspose));
        }
        {
            // C = MatMul(B, A)
            std::shared_ptr<Tensor> C(new Tensor);
            C->buffer().type       = halide_type_of<float>();
            C->buffer().dimensions = 2;
            C->setLength(0, ic * kw * kh);
            C->setLength(1, oc);
            auto cmd = GeometryComputerUtils::makeMatMul(B, A, C.get(), nullptr, false, true);
            auto kernelDiffDes        = TensorUtils::getDescribe(outputs[0]);
            kernelDiffDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

            // Transpose
            auto len0 = kw * kh * ic;
            auto len1 = oc;
            kernelDiffDes->regions.resize(1);
            auto& desReg         = kernelDiffDes->regions[0];
            desReg.size[0]       = 1;
            desReg.size[1]       = len1;
            desReg.size[2]       = len0;
            desReg.dst.offset    = 0;
            desReg.dst.stride[0] = 0;
            desReg.dst.stride[1] = len0;
            desReg.dst.stride[2] = 1;
            desReg.src.offset    = 0;
            desReg.src.stride[0] = 0;
            desReg.src.stride[1] = 1;
            desReg.src.stride[2] = len1;
            desReg.origin        = C.get();
            res.extras.emplace_back(std::move(C));
            res.command.emplace_back(std::move(cmd));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConv2DBackPropFilter);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Conv2DBackPropFilter});
}

REGISTER_GEOMETRY(GeometryConv2DBackPropFilter, _create);

} // namespace MNN
