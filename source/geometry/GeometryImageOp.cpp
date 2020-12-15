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
static void _ConverterInterp(const Interp* resize, InterpT* dstInfo, int inW, int inH, int outW, int outH) {
    switch (resize->ctm()) {
        case CoordinateTransformationMode_NotSet:
        {
            // For compability, old model's nearest don't support halfpixels
            if (resize->halfPixelCenters() && resize->resizeType() != 1) {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            } else if (resize->alignCorners()) {
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
            } else {
                dstInfo->heightScale = (float)(inH) / (float)(outH);
                dstInfo->widthScale  = (float)(inW) / (float)(outW);
            }
            break;
        }
        case CoordinateTransformationMode_AlignCorners:
        {
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
            break;
        }
        case CoordinateTransformationMode_HalfPixels:
        {
            dstInfo->heightScale = (float)(inH) / (float)(outH);
            dstInfo->widthScale  = (float)(inW) / (float)(outW);
            dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            break;
        }
        case CoordinateTransformationMode_PytorchHalfPixels:
        {
            if (outH > 1) {
                dstInfo->heightScale = (float)inH / (float)outH;
                dstInfo->heightOffset = 0.5f * dstInfo->heightScale - 0.5f;
            } else {
                dstInfo->heightScale = 0.0f;
            }
            if (outW > 1) {
                dstInfo->widthScale = (float)inW / (float)outW;
                dstInfo->widthOffset = 0.5f * dstInfo->widthScale - 0.5f;
            } else {
                dstInfo->widthScale = 0.0f;
            }
            break;
        }
        case CoordinateTransformationMode_Asymmetric:
        {
            dstInfo->heightScale = (float)(inH) / (float)(outH);
            dstInfo->widthScale  = (float)(inW) / (float)(outW);
            break;
        }
        case CoordinateTransformationMode_TensorflowHalfPixels:
        {
            dstInfo->heightScale = (float)(inH) / (float)(outH);
            dstInfo->widthScale  = (float)(inW) / (float)(outW);
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
            std::unique_ptr<OpT> interp(new OpT);
            if (nullptr != op->name()) {
                interp->name = op->name()->str();
            }
            interp->type                          = OpType_Interp;
            interp->main.type                     = OpParameter_Interp;
            interp->main.value                    = new InterpT;
            interp->main.AsInterp()->widthScale = (float)inputs[0]->width() / (float)outputs[0]->width();
            interp->main.AsInterp()->heightScale = (float)inputs[0]->height() / (float)outputs[0]->height();
            interp->main.AsInterp()->resizeType   = 2; // bilinear
            res.command.emplace_back(GeometryComputerUtils::makeCommand(interp.get(), newInputs, newOutputs));
        }
        else if (OpType_Interp == op->type()) {
            // Compute cord transform for interp
            std::unique_ptr<OpT> interp(new OpT);
            if (nullptr != op->name()) {
                interp->name = op->name()->str();
            }
            interp->type                          = OpType_Interp;
            auto resize                           = op->main_as_Interp();
            interp->main.type                     = OpParameter_Interp;
            interp->main.value                    = new InterpT;
            interp->main.AsInterp()->resizeType   = resize->resizeType();
            auto inW = inputs[0]->width();
            auto inH = inputs[0]->height();
            auto outW = outputs[0]->width();
            auto outH = outputs[0]->height();
            auto dstInfo = interp->main.AsInterp();
            _ConverterInterp(resize, dstInfo, inW, inH, outW, outH);
            res.command.emplace_back(GeometryComputerUtils::makeCommand(interp.get(), newInputs, newOutputs));
        } else {
            Command cmd;
            cmd.op      = op;
            cmd.inputs  = std::move(newInputs);
            cmd.outputs = std::move(newOutputs);
            res.command.emplace_back(std::move(cmd));
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
