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
    virtual bool onRecompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
            Context& context, CommandBuffer& res) const override {
        return false;
    }
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        // Origin convolution with format converter
        return GeometryConvUtils::computeSingle(op, inputs, outputs, context, res);
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
        int pl,pt,pr,pb;
        if (common->pads() == nullptr) {
            pl = common->padX();
            pr = common->padX();
            pt = common->padY();
            pb = common->padY();
        } else {
            pl = common->pads()->data()[1];
            pr = common->pads()->data()[3];
            pt = common->pads()->data()[0];
            pb = common->pads()->data()[2];
        }
        auto batch = input->batch();
        auto ic    = input->channel();
        auto iw    = input->width();
        auto ih    = input->height();
        auto pads  = std::make_pair(pl, pt);
        auto ow    = (iw + pl + pr - kw) / sw + 1;
        auto oh    = (ih + pt + pb - kh) / sh + 1;
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
        int pl,pt,pr,pb;
        if (common->pads() == nullptr) {
            pl = common->padX();
            pr = common->padX();
            pt = common->padY();
            pb = common->padY();
        } else {
            pl = common->pads()->data()[1];
            pr = common->pads()->data()[3];
            pt = common->pads()->data()[0];
            pb = common->pads()->data()[2];
        }
        auto batch = output->batch();
        auto ic    = output->channel();
        auto iw    = output->width();
        auto ih    = output->height();
        auto pads  = std::make_pair(pl, pt);
        auto ow    = (iw + pl + pr - kw) / sw + 1;
        auto oh    = (ih + pt + pb - kh) / sh + 1;
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
