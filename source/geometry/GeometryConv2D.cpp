//
//  GeometryConv2D.cpp
//  MNN
//
//  Created by MNN on 2020/07/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <limits>
#include "ConvertUtils.hpp"
#include "GeometryConvUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {

class GeometryConv2D : public DefaultGeometryComputer {
public:
    // Im2Col + GEMM
    bool computeIm2Col_GEMM(  const Convolution2DCommon* common, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            Context& context, CommandBuffer& res) const {
        auto input      = inputs[0];
        auto outputDiff = outputs[0];
        MNN_ASSERT(1 == common->group());
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
            GeometryConvUtils::im2Col(im2Col.get(), input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
            B = im2Col.get();
            res.extras.emplace_back(im2Col);
        }
        {
            // A: Weight oc, ic, kh, kw -> oc, ic*kh*kw
            std::shared_ptr<Tensor> kernel(new Tensor);
            A                           = kernel.get();
            kernel->buffer().type       = halide_type_of<float>();
            kernel->buffer().dimensions = 2;
            kernel->setLength(0, oc);
            kernel->setLength(1, ic * kw * kh);
            auto des             = TensorUtils::getDescribe(kernel.get());
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            GeometryComputerUtils::makeRawAddressRef(kernel.get(), inputs[1], 0, ic * kw * kh * oc);
            res.extras.emplace_back(std::move(kernel));
        }
        {
            // C = MatMul(B, A)
            std::shared_ptr<Tensor> C(new Tensor);
            C->buffer().type       = halide_type_of<float>();
            C->buffer().dimensions = 2;
            C->setLength(0, batch * ow * oh);
            C->setLength(1, oc);
            TensorUtils::getDescribe(C.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            Tensor* bias                                       = nullptr;
            if (inputs.size() > 2) {
                bias = inputs[2];
            }
            res.command.emplace_back(GeometryComputerUtils::makeMatMul(B, A, C.get(), bias, true, true));
            res.extras.emplace_back(C);

            // Activation
            float minValue = 0.0f, maxValue = 6.0f;
            bool needPostTreat = false;
            if (common->relu()) {
                needPostTreat = true;
                minValue      = 0.0f;
                maxValue      = std::numeric_limits<float>().max();
            }
            if (common->relu6()) {
                needPostTreat = true;
                minValue      = 0.0f;
                maxValue      = 6.0f;
            }
            if (needPostTreat) {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(GeometryConvUtils::makeRelu6(builder, minValue, maxValue));
                std::shared_ptr<Tensor> C2(new Tensor);
                C2->buffer().type       = halide_type_of<float>();
                C2->buffer().dimensions = 2;
                C2->setLength(0, batch * ow * oh);
                C2->setLength(1, oc);
                TensorUtils::getDescribe(C2.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                auto cmd = GeometryComputerUtils::makeCommand(builder, {C.get()}, {C2.get()});
                res.command.emplace_back(cmd);
                res.extras.emplace_back(C2);
                C = C2;
            }
            // Transpose
            // Batch, oh, ow, oc -> batch, oc, oh, ow
            TensorUtils::setLinearLayout(C.get());
            if (ow == oh && oh == 1) {
                GeometryComputerUtils::makeRawAddressRef(outputs[0], C.get(), 0, batch * oc);
            } else {
                auto kernelDiffDes        = TensorUtils::getDescribe(outputs[0]);
                kernelDiffDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                kernelDiffDes->regions.resize(1);
                auto& desReg         = kernelDiffDes->regions[0];
                desReg.size[0]       = batch;
                desReg.size[1]       = oc;
                desReg.size[2]       = oh * ow;
                desReg.dst.offset    = 0;
                desReg.dst.stride[0] = oc * oh * ow;
                desReg.dst.stride[1] = oh * ow;
                desReg.dst.stride[2] = 1;
                desReg.src.offset    = 0;
                desReg.src.stride[0] = oh * ow * oc;
                desReg.src.stride[1] = 1;
                desReg.src.stride[2] = oc;
                desReg.origin        = C.get();
            }
        }
        return true;
    }
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            Context& context, CommandBuffer& res) const override {
        return false;
    }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (inputs.size() == 1) {
            // Origin convolution with format converter
            return GeometryConvUtils::computeSingle(op, inputs, outputs, context, res);
        }
        auto common = op->main_as_Convolution2D()->common();
        if (common->outputCount() > 0) {
            // FIXME: Remove this logical in future
            if (context.forwardType() == MNN_FORWARD_CPU || context.forwardType() == MNN_FORWARD_CPU_EXTENSION || context.forwardType() == MNN_FORWARD_OPENCL) {
                auto inputDes     = TensorUtils::getDescribe(inputs[0]);
                auto format       = inputDes->dimensionFormat;
                if (MNN_DATA_FORMAT_NC4HW4 == format) {
                    return DefaultGeometryComputer::onCompute(op, inputs, outputs, context, res);
                }
            }
            return computeIm2Col_GEMM(common, inputs, outputs, context, res);
        }
        std::unique_ptr<Convolution2DCommonT> temp(common->UnPack());
        temp->outputCount = inputs[1]->length(0);
        temp->kernelY = inputs[1]->length(2);
        temp->kernelX = inputs[1]->length(3);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(Convolution2DCommon::Pack(builder, temp.get()));
        return computeIm2Col_GEMM(flatbuffers::GetRoot<MNN::Convolution2DCommon>(builder.GetBufferPointer()), inputs, outputs, context, res);
    }
};


class GeometryConvTranspose2D : public GeometryConv2D {
public:
    // Im2Col + GEMM
    bool computeGEMM_Col2Im(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                            Context& context, CommandBuffer& res) const {
        auto common     = op->main_as_Convolution2D()->common();
        auto input      = inputs[0];
        auto outputDiff = outputs[0];
        auto weight = inputs[1];
        MNN_ASSERT(1 == common->group());
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
        auto pads  = ConvolutionCommon::convolutionTransposePad(input, outputDiff, common);
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        MNN_ASSERT(TensorUtils::getDescribe(outputDiff)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        Tensor* A = nullptr;
        Tensor* B = nullptr;
        {
            // B: Input n, ic, ih, iw -> ic, n * ih * iw
            std::shared_ptr<Tensor> dest(Tensor::createDevice<float>({ic, batch * ih * iw}));
            res.extras.emplace_back(dest);
            B = dest.get();
            auto des = TensorUtils::getDescribe(dest.get());
            des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->regions.resize(1);
            auto& reg = des->regions[0];
            reg.origin = input;
            reg.size[0] = ic;
            reg.size[1] = batch;
            reg.size[2] = ih * iw;
            reg.src.offset = 0;
            reg.src.stride[0] = ih * iw;
            reg.src.stride[1] = ic * ih * iw;
            reg.src.stride[2] = 1;
            reg.dst.offset = 0;
            reg.dst.stride[0] = ih * iw * batch;
            reg.dst.stride[1] = ih * iw;
            reg.dst.stride[2] = 1;
        }
        {
            // A: Weight ic, oc, kh, kw -> ic, oc*kh*kw
            std::shared_ptr<Tensor> kernel(Tensor::createDevice<float>({ic, oc * kw * kh}));
            A                           = kernel.get();
            GeometryComputerUtils::makeRawAddressRef(kernel.get(), weight, 0, ic * kw * kh * oc);
            res.extras.emplace_back(std::move(kernel));
        }
        {
            // C = MatMul(B, A)
            std::shared_ptr<Tensor> C(Tensor::createDevice<float>({oc * kw * kh, batch * ih * iw}));
            res.command.emplace_back(GeometryComputerUtils::makeMatMul(A, B, C.get(), nullptr, true, false));
            res.extras.emplace_back(C);

            // Col2Im:
            // 1. C-> C' batch, oc, oh, ow, kw*kh, 2. C' -> C'' batch, oc, oh, ow (reduce_sum)
            // 3. C'' -> C'' + bias, 4. posttreat(C'' + bias)
            std::shared_ptr<Tensor> C_(Tensor::createDevice<float>({1, kw * kh, batch * oc * oh * ow}));
            res.extras.emplace_back(C_);
            {
                std::shared_ptr<Tensor> im2ColTemp(Tensor::createDevice<float>({oc * kw * kh, batch * ih * iw}));
                // Swap ow, iw, oh, ih for im2Col
                GeometryConvUtils::im2Col(im2ColTemp.get(), outputDiff, oc, kh, kw, batch, ih, iw, oh, ow, sh, sw, dh, dw, pads, oh * ow * oc * batch);
                auto des = TensorUtils::getDescribe(C_.get());
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                auto originDes = TensorUtils::getDescribe(im2ColTemp.get());
                des->regions = std::move(originDes->regions);
                // Swap src and dst, from im2col->col2im
                for (auto& reg : des->regions) {
                    reg.origin = C.get();
                    auto temp = reg.src;
                    reg.src = std::move(reg.dst);
                    reg.dst = std::move(temp);
                }
            }
            std::shared_ptr<Tensor> C__(Tensor::createDevice<float>({1, 1, batch * oc * oh * ow}));
            res.extras.emplace_back(C__);
            res.command.emplace_back(GeometryComputerUtils::makeReduce(ReductionType_SUM, C_.get(), C__.get()));

            if (inputs.size() > 2) {
                MNN_ASSERT(oc == inputs[2]->elementSize());
                std::shared_ptr<Tensor> biasLarge(Tensor::createDevice<float>({1, 1, batch * oc * oh * ow}));
                res.extras.emplace_back(biasLarge);
                auto des = TensorUtils::getDescribe(biasLarge.get());
                des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->regions.resize(1);
                auto& reg = des->regions[0];
                reg.origin = inputs[2];
                reg.size[0] = batch;
                reg.size[1] = oc;
                reg.size[2] = oh * ow;
                reg.src.offset = 0;
                reg.src.stride[0] = 0;
                reg.src.stride[1] = 1;
                reg.src.stride[2] = 0;
                reg.dst.offset = 0;
                reg.dst.stride[0] = oc * oh * ow;
                reg.dst.stride[1] = oh * ow;
                reg.dst.stride[2] = 1;
                std::shared_ptr<Tensor> temp(Tensor::createDevice<float>({1, 1, batch * oh * ow * oc}));
                res.extras.emplace_back(temp);
                res.command.emplace_back(GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, C__.get(), biasLarge.get(), temp.get()));
                C__ = temp;
            }

            // Activation
            float minValue = 0.0f, maxValue = 0.0f;
            bool needPostTreat = false;
            if (common->relu()) {
                needPostTreat = true;
                minValue      = 0.0f;
                maxValue      = std::numeric_limits<float>().max();
            }
            if (common->relu6()) {
                needPostTreat = true;
                minValue      = 0.0f;
                maxValue      = 6.0f;
            }
            if (needPostTreat) {
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(GeometryConvUtils::makeRelu6(builder, minValue, maxValue));
                std::shared_ptr<Tensor> C2(new Tensor);
                C2->buffer().type       = halide_type_of<float>();
                C2->buffer().dimensions = 3;
                C2->setLength(0, 1);
                C2->setLength(1, 1);
                C2->setLength(2, batch * ow * oh * oc);
                TensorUtils::getDescribe(C2.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                auto cmd = GeometryComputerUtils::makeCommand(builder, {C__.get()}, {C2.get()});
                res.command.emplace_back(cmd);
                res.extras.emplace_back(C2);
                C__ = C2;
            }
            GeometryComputerUtils::makeRawAddressRef(outputs[0], C__.get(), 0, oc * batch * ow * oh);
        }
        return true;
    }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (op->main_as_Convolution2D()->common()->hasOutputShape()) {
            const std::vector<Tensor*> newInputs(inputs.begin(), inputs.end() - 1);
            if (newInputs.size() == 1) {
                // Origin convolution with format converter
                return GeometryConvUtils::computeSingle(op, newInputs, outputs, context, res);
            }
            return computeGEMM_Col2Im(op, newInputs, outputs, context, res);
        } else {
            if (inputs.size() == 1) {
                // Origin convolution with format converter
                return GeometryConvUtils::computeSingle(op, inputs, outputs, context, res);
            }
            return computeGEMM_Col2Im(op, inputs, outputs, context, res);
        }
    }
};
class GeometryIm2Col : public GeometryConv2D {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto common = op->main_as_Convolution2D()->common();
        auto input  = inputs[0];
        auto output = outputs[0];
        auto kw    = common->kernelX();
        auto kh    = common->kernelY();
        auto sw    = common->strideX();
        auto sh    = common->strideY();
        auto dw    = common->dilateX();
        auto dh    = common->dilateY();
        auto pw    = common->padX();
        auto ph    = common->padY();
        auto batch = input->batch();
        auto ic    = input->channel();
        auto iw    = input->width();
        auto ih    = input->height();
        auto pads  = std::make_pair(pw, ph);
        auto ow    = (iw + pw * 2 - kw) / sw + 1;
        auto oh    = (ih + ph * 2 - kh) / sh + 1;
        GeometryConvUtils::im2Col(output, input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConv2D);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Convolution});

    std::shared_ptr<GeometryComputer> comp2(new GeometryConvTranspose2D);
    GeometryComputer::registerGeometryComputer(comp2, {OpType_Deconvolution});

    std::shared_ptr<GeometryComputer> comp3(new GeometryIm2Col);
    GeometryComputer::registerGeometryComputer(comp3, {OpType_Im2Col});
}

REGISTER_GEOMETRY(GeometryConv2D, _create);

} // namespace MNN
