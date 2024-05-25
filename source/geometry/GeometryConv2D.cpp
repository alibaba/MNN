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
            auto tmpT = GeometryConvUtils::im2Col(im2Col.get(), input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
            if (nullptr != tmpT) {
                res.extras.emplace_back(tmpT);
            }
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
            if (context.forwardType() == MNN_FORWARD_CPU || context.forwardType() == MNN_FORWARD_CPU_EXTENSION || context.forwardType() == MNN_FORWARD_OPENCL ||
                context.forwardType() == MNN_FORWARD_VULKAN
                ) {
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
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        if (op->main_as_Convolution2D()->common()->hasOutputShape()) {
            const std::vector<Tensor*> newInputs(inputs.begin(), inputs.end() - 1);
            // Origin convolution with format converter
            return GeometryConvUtils::computeSingle(op, newInputs, outputs, context, res);
        }
        // Origin convolution with format converter
        return GeometryConvUtils::computeSingle(op, inputs, outputs, context, res);
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
        auto tmpT = GeometryConvUtils::im2Col(output, input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
        if (nullptr != tmpT) {
            res.extras.emplace_back(tmpT);
        }
        return true;
    }
};

class GeometryCol2Im : public GeometryConv2D {
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
        auto batch = output->batch();
        auto ic    = output->channel();
        auto iw    = output->width();
        auto ih    = output->height();
        auto pads  = std::make_pair(pw, ph);
        auto ow    = (iw + pw * 2 - kw) / sw + 1;
        auto oh    = (ih + ph * 2 - kh) / sh + 1;
        auto shape = output->shape();
        auto ishape = input->shape();
        int n = ishape[0];
        int ickhkw = ishape[1];
        int ohow = ishape[2];
        // set batch = 1, then loopNumber = batch
        auto tmpIm2Col = GeometryConvUtils::im2Col(output, input, ic, kh, kw, 1, oh, ow, ih, iw, sh, sw, dh, dw, pads);
        if (nullptr != tmpIm2Col) {
            res.extras.emplace_back(tmpIm2Col);
        }
        auto des = TensorUtils::getDescribe(output);
        // build cmd
        flatbuffers::FlatBufferBuilder builder;
        OpBuilder bianryOp(builder);
        bianryOp.add_type(OpType_UnaryOp);
        auto bianryOpOffset = bianryOp.Finish();
        auto iterIndexesOffset = builder.CreateVector(std::vector<int>{-1, -1});
        auto stepOffset = builder.CreateVector(std::vector<int>{ic*iw*ih, ickhkw*ohow});
        auto indexesOffset = builder.CreateVector(std::vector<int>{1, 0});
        std::vector<flatbuffers::Offset<RegionCommand>> rcmdAllOffset;
        for (auto& region : des->regions) {
            auto tmp   = region.dst;
            region.dst = region.src;
            region.src = tmp;
            //size
            auto sizeOffset = builder.CreateVector(std::vector<int>{region.size[0], region.size[1], region.size[2]});
            // View 0 - dst
            auto view0Stride = builder.CreateVector(std::vector<int>{region.dst.stride[0], region.dst.stride[1], region.dst.stride[2]});
            ViewBuilder view0Builder(builder);
            view0Builder.add_offset(region.dst.offset);
            view0Builder.add_stride(view0Stride);
            auto view0Offset = view0Builder.Finish();
            // View 1 - src
            auto view1Stride = builder.CreateVector(std::vector<int>{region.src.stride[0], region.src.stride[1], region.src.stride[2]});
            ViewBuilder view1Builder(builder);
            view1Builder.add_offset(region.src.offset);
            view1Builder.add_stride(view1Stride);
            auto view1Offset = view1Builder.Finish();
            auto viewAllOffset = builder.CreateVector<flatbuffers::Offset<View>>({view0Offset, view1Offset});
            RegionCommandBuilder rcmdBuild(builder);
            rcmdBuild.add_op(bianryOpOffset);
            rcmdBuild.add_view(viewAllOffset);
            rcmdBuild.add_indexes(indexesOffset);
            rcmdBuild.add_iterIndexes(iterIndexesOffset);
            rcmdBuild.add_steps(stepOffset);
            rcmdBuild.add_size(sizeOffset);
            rcmdBuild.add_fuse(BinaryOpOperation_ADD); // zreduce add
            rcmdAllOffset.push_back(rcmdBuild.Finish());
        }
        auto rcmdAllOffsets = builder.CreateVector<flatbuffers::Offset<RegionCommand>>(rcmdAllOffset);
        auto inputIndexesOffset = builder.CreateVector(std::vector<int>{0});
        auto outputIndexesOffset = builder.CreateVector(std::vector<int>{1});
        // view0 and view1 is the same
        RegionCommandBuilder initrcmdBuild(builder);
        initrcmdBuild.add_indexes(outputIndexesOffset);
        auto initrcmdOffset = initrcmdBuild.Finish();
        auto initrcmdOffsetMulti = builder.CreateVector<flatbuffers::Offset<RegionCommand>>({initrcmdOffset});
        std::vector<flatbuffers::Offset<RegionCommand>> initCommandOffsets;
        initCommandOffsets.emplace_back(initrcmdOffset);
        LoopParamBuilder loopBuilder(builder);
        loopBuilder.add_initCommand(initrcmdOffsetMulti);
        loopBuilder.add_commands(rcmdAllOffsets);
        loopBuilder.add_loopNumber(batch);
        loopBuilder.add_tensorNumber(2);
        loopBuilder.add_parallel(true);
        loopBuilder.add_inputIndexes(inputIndexesOffset);
        loopBuilder.add_outputIndexes(outputIndexesOffset);
        auto loopOffset = loopBuilder.Finish();
        flatbuffers::Offset<flatbuffers::String> nameOffset;
        if (nullptr != op->name()) {
            nameOffset = builder.CreateString(op->name()->c_str());
        }
        OpBuilder finishBuilder(builder);
        finishBuilder.add_main(loopOffset.Union());
        finishBuilder.add_main_type(OpParameter_LoopParam);
        finishBuilder.add_type(OpType_While);
        if (nullptr != op->name()) {
            finishBuilder.add_name(nameOffset);
        }
        builder.Finish(finishBuilder.Finish());
        auto cmd = GeometryComputerUtils::makeCommand(builder, {inputs[0]}, outputs);
        res.command.emplace_back(std::move(cmd));

        des->regions.clear();
        TensorUtils::getDescribe(output)->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
        output->buffer().dimensions = shape.size();
        for (int i = 0; i < shape.size(); i++) {
            output->setLength(i, shape[i]);
        }
        TensorUtils::setLinearLayout(output);
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

    std::shared_ptr<GeometryComputer> comp4(new GeometryCol2Im);
    GeometryComputer::registerGeometryComputer(comp4, {OpType_Col2Im});
}

REGISTER_GEOMETRY(GeometryConv2D, _create);

} // namespace MNN
