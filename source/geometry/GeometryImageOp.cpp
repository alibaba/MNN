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


/**
 if coordinate_transformation_mode is "half_pixel",
 x_original = (x_resized + 0.5) / scale - 0.5,

 if coordinate_transformation_mode is "pytorch_half_pixel",
 x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,

 if coordinate_transformation_mode is "align_corners",
 x_original = x_resized * (length_original - 1) / (length_resized - 1),

 if coordinate_transformation_mode is "asymmetric",
 x_original = x_resized / scale,

 if coordinate_transformation_mode is "tf_half_pixel_for_nn",
 x_original = (x_resized + 0.5) / scale,

 if coordinate_transformation_mode is "tf_crop_and_resize",
 x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).
 */
struct InterpInfo {
    float heightScale;
    float widthScale;
    float widthOffset = 0.0f;
    float heightOffset = 0.0f;
};
static void _ConverterInterp(const Interp* resize, InterpInfo* dstInfo, int inW, int inH, int outW, int outH, bool computeScale = true) {
    switch (resize->ctm()) {
        case CoordinateTransformationMode_NotSet:
        {
            // For compability, old model's nearest don't support halfpixels
            if (resize->halfPixelCenters() && resize->resizeType() != 1) {
                if (computeScale) {
                    dstInfo->heightScale = (float)(inH) / (float)(outH);
                    dstInfo->widthScale  = (float)(inW) / (float)(outW);
                }
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            } else if (resize->alignCorners()) {
                if (computeScale) {
                    if (outH == 1) {
                        dstInfo->heightScale = 0.0f;
                    } else {
                        dstInfo->heightScale = (float)(inH - 1) / (float)(outH - 1);
                    }
                    if (outW == 1) {
                        dstInfo->widthScale = 0.0f;
                    } else {
                        dstInfo->widthScale  = (float)(inW - 1) / (float)(outW - 1);
                    }
                }
            } else if (computeScale) {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            break;
        }
        case CoordinateTransformationMode_AlignCorners:
        {
            if (computeScale) {
                if (outH == 1) {
                    dstInfo->heightScale = 0.0f;
                } else {
                    dstInfo->heightScale = (float)(inH - 1) / (float)(outH - 1);
                }
                if (outW == 1) {
                    dstInfo->widthScale = 0.0f;
                } else {
                    dstInfo->widthScale  = (float)(inW - 1) / (float)(outW - 1);
                }
            }
            break;
        }
        case CoordinateTransformationMode_HalfPixels:
        {
            if (computeScale) {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            break;
        }
        case CoordinateTransformationMode_PytorchHalfPixels:
        {
            if (outH > 1) {
                if (computeScale) {
                    dstInfo->heightScale = (float)inH / (float)outH;
                }
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            } else {
                if (computeScale) {
                    dstInfo->heightScale = 0.0f;
                }
            }
            if (outW > 1) {
                if (computeScale) {
                    dstInfo->widthScale = (float)inW / (float)outW;
                }
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            } else {
                if (computeScale) {
                    dstInfo->widthScale = 0.0f;
                }
            }
            break;
        }
        case CoordinateTransformationMode_Asymmetric:
        {
            if (computeScale) {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            break;
        }
        case CoordinateTransformationMode_TensorflowHalfPixels:
        {
            if (computeScale) {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            dstInfo->widthOffset = 0.5f * dstInfo->widthScale;
            dstInfo->heightOffset = 0.5f * dstInfo->heightScale;
            break;
        }
        case CoordinateTransformationMode_TensorflowCropAndResize:
        {
            //FIXME: Not support now
            MNN_ERROR("Don't support CoordinateTransformationMode_TensorflowCropAndResize currently\n");
            break;
        }
        default:
            break;
    }
}
static flatbuffers::Offset<Op> makeInterp(flatbuffers::FlatBufferBuilder& builder, const InterpInfo* info, int resizeType, const Op* op) {
    flatbuffers::Offset<flatbuffers::String> temp;
    if (nullptr != op->name()) {
        temp = builder.CreateString(op->name()->str());
    }
    InterpBuilder intpB(builder);
    intpB.add_resizeType(resizeType);
    intpB.add_widthScale(info->widthScale);
    intpB.add_heightScale(info->heightScale);
    intpB.add_heightOffset(info->heightOffset);
    intpB.add_widthOffset(info->widthOffset);
    auto offsetInterp = intpB.Finish().Union();
    OpBuilder opB(builder);
    opB.add_type(OpType_Interp);
    opB.add_main(offsetInterp);
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
            builder.Finish(makeInterp(builder, &info, 2, op));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {newInputs[0]}, newOutputs));
        }
        else if (OpType_Interp == op->type()) {
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
            _ConverterInterp(resize, &info, inW, inH, outW, outH, computeScale);
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(makeInterp(builder, &info, resize->resizeType(), op));
            res.command.emplace_back(GeometryComputerUtils::makeCommand(builder, {newInputs[0]}, newOutputs));
        } else {
            SharedPtr<Command> cmdP = new Command;
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
        OpType_Resize,
        OpType_Int8ToFloat,
        OpType_FloatToInt8
    });
}

REGISTER_GEOMETRY(GeometryImageOp, _create);

} // namespace MNN
