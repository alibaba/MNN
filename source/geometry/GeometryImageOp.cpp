//
//  GeometryImageOp.cpp
//  MNN
//
//  Created by MNN on 2020/05/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "shape/SizeComputer.hpp"
namespace MNN {

static flatbuffers::Offset<Op> makeInterp(flatbuffers::FlatBufferBuilder& builder, const InterpInfo* info, int resizeType, const Op* op, OpType type) {
    flatbuffers::Offset<flatbuffers::String> temp;
    if (nullptr != op->name()) {
        temp = builder.CreateString(op->name()->str());
    }
    InterpBuilder intp3DB(builder);
    intp3DB.add_resizeType(resizeType);
    intp3DB.add_widthScale(info->widthScale);
    intp3DB.add_heightScale(info->heightScale);
    intp3DB.add_depthScale(info->depthScale);
    intp3DB.add_heightOffset(info->heightOffset);
    intp3DB.add_widthOffset(info->widthOffset);
    intp3DB.add_depthOffset(info->depthOffset);
    auto offsetInterp3D = intp3DB.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(type);
    opB.add_main(offsetInterp3D);
    opB.add_main_type(OpParameter_Interp);
    if (nullptr != op->name()) {
        opB.add_name(temp);
    }
    return opB.Finish();
}

class GeometryImageOp : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
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
        if (OpType_Resize == op->type()) {
            // Turn resize to interp
            InterpInfo info;
            info.widthScale = (float)inputs[0]->width() / (float)outputs[0]->width();
            info.heightScale = (float)inputs[0]->height() / (float)outputs[0]->height();
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(makeInterp(builder, &info, 2, op, OpType_Interp));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {newInputs[0]}, newOutputs));
        }
        else if (OpType_Interp == op->type() && inputs[0]->dimensions() <= 4) {
            // Compute cord transform for interp
            auto resize                           = op->main_as_Interp();
            auto inW = inputs[0]->width();
            auto inH = inputs[0]->height();
            auto outW = outputs[0]->width();
            auto outH = outputs[0]->height();
            InterpInfo info;
            bool computeScale = true;
            if (inputs.size() > 1 && inputs[1]->getType().code == halide_type_float) {
                computeScale = false;
                info.heightScale = 1.0f / inputs[1]->host<float>()[2];
                if (inputs[0]->dimensions() >= 4) {
                    info.widthScale = 1.0f / inputs[1]->host<float>()[3];
                }
            }
            const int defaultDepth = 10;
            _ConverterInterp(resize, &info, inW, inH, defaultDepth, outW, outH, defaultDepth, computeScale);
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(makeInterp(builder, &info, resize->resizeType(), op, OpType_Interp));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {newInputs[0]}, newOutputs));
        } else if (OpType_Interp == op->type() && inputs[0]->dimensions() == 5) {
            // Compute cord transform for interp
            auto resize                           = op->main_as_Interp();
            auto inShape = newInputs[0]->shape();
            auto outShape = newOutputs[0]->shape();
            auto inW = inShape[4];
            auto inH = inShape[3];
            auto inD = inShape[2];
            auto outW = outShape[4];
            auto outH = outShape[3];
            auto outD = outShape[2];
            InterpInfo info;
            bool computeScale = true;
            if (inputs.size() > 1 && inputs[1]->getType().code == halide_type_float) {
                computeScale = false;
                info.depthScale = 1.0f / inputs[1]->host<float>()[2];
                info.heightScale = 1.0f / inputs[1]->host<float>()[3];
                info.widthScale = 1.0f / inputs[1]->host<float>()[4];
            }
            _ConverterInterp(resize, &info, inW, inH, inD, outW, outH, outD, computeScale);
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(makeInterp(builder, &info, resize->resizeType(), op, OpType_Interp3D));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {newInputs[0]}, newOutputs));
        } else {
            std::shared_ptr<Command> cmdP(new Command);
            auto& cmd = *cmdP;;
            cmd.op      = op;
            cmd.inputs  = std::move(newInputs);
            cmd.outputs = std::move(newOutputs);
            res.command.emplace_back(std::move(cmdP));
        }
        if (originOutput != output) {
            ConvertUtils::compute(output, originOutput, res);
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryImageOp);
    GeometryComputer::registerGeometryComputer(
        comp, {
        OpType_ConvInt8,
        OpType_DepthwiseConvInt8,
        OpType_ConvolutionDepthwise,
        OpType_DeconvolutionDepthwise,
        OpType_Pooling,
        OpType_Interp,
        OpType_Interp3D,
        OpType_Resize,
        OpType_Int8ToFloat,
        OpType_FloatToInt8
    });
}

REGISTER_GEOMETRY(GeometryImageOp, _create);

} // namespace MNN
