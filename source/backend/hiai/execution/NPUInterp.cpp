//
//  NPUInterp.cpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NPUInterp.hpp"
#include "NPUBackend.hpp"

using namespace std;

namespace MNN {
namespace {

// The bundled HiAI headers omit this declaration even though the 3.6.1
// backend emits the operator. Keep the compatibility declaration local so
// the vendor header ABI remains byte-identical to the upstream baseline.
class ResizeBicubicCompat final : public ge::Operator {
public:
    explicit ResizeBicubicCompat(const std::string& name) : ge::Operator(name, "ResizeBicubic", 6) {
        InputRegister("x");
        InputRegister("size");
        OutputRegister("y");
        OptionalAttrRegister("align_corners", ge::AttrValue::CreateFrom(false));
        OptionalAttrRegister("half_pixel_centers", ge::AttrValue::CreateFrom(false));
    }

    ResizeBicubicCompat& set_input_x(const ge::Operator& input) {
        SetInput("x", input);
        return *this;
    }
    ResizeBicubicCompat& set_input_size(const ge::Operator& input) {
        SetInput("size", input);
        return *this;
    }
    ResizeBicubicCompat& set_attr_align_corners(bool value) {
        SetAttr("align_corners", ge::AttrValue::CreateFrom(value));
        return *this;
    }
    ResizeBicubicCompat& set_attr_half_pixel_centers(bool value) {
        SetAttr("half_pixel_centers", ge::AttrValue::CreateFrom(value));
        return *this;
    }
};

} // namespace

NPUInterp::NPUInterp(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                     const std::vector<MNN::Tensor*>& outputs)
    : NPUCommonExecution(b, op) {}

ErrorCode NPUInterp::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mNpuBackend->setNetworkInput(inputs, mOp);

    if (!mNpuBackend->isExplicitHiAISession()) {
        auto opName = mOp->name()->str();
        auto param = mOp->main_as_Interp();
        auto xOp = mNpuBackend->getInputOps(mOp);
        auto resizeType = param->resizeType();
        MNN_ASSERT(resizeType <= 3);
        if (resizeType > 3) {
            MNN_ERROR("npu Interp not support type: %d", resizeType);
            return NOT_SUPPORT;
        }
        vector<int32_t> hw = {outputs[0]->height(), outputs[0]->width()};
        mConstShape = hiai::op::Const(opName + "_w_const");
        {
            ge::TensorDesc fdesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32);
            ge::TensorPtr filter = std::make_shared<ge::Tensor>();
            filter->SetTensorDesc(fdesc);
            filter->SetData((uint8_t*)hw.data(), hw.size() * sizeof(int32_t));
            mConstShape.set_attr_value(filter);
        }
        if (resizeType == 1) {
            shared_ptr<hiai::op::ResizeNearestNeighborV2> interp(new hiai::op::ResizeNearestNeighborV2(opName));
            (*interp)
                .set_input_x(*xOp)
                .set_input_size(mConstShape)
                .set_attr_align_corners(param->alignCorners())
                .set_attr_half_pixel_centers(param->halfPixelCenters());
            mNpuBackend->setOutputOps(mOp, {interp}, outputs);
        } else if (resizeType == 2) {
            shared_ptr<hiai::op::ResizeBilinearV2> interp(new hiai::op::ResizeBilinearV2(opName));
            (*interp)
                .set_input_x(*xOp)
                .set_input_size(mConstShape)
                .set_attr_align_corners(param->alignCorners())
                .set_attr_half_pixel_centers(param->halfPixelCenters());
            mNpuBackend->setOutputOps(mOp, {interp}, outputs);
        } else if (resizeType == 3) {
            shared_ptr<ResizeBicubicCompat> interp(new ResizeBicubicCompat(opName));
            (*interp)
                .set_input_x(*xOp)
                .set_input_size(mConstShape)
                .set_attr_align_corners(param->alignCorners())
                .set_attr_half_pixel_centers(param->halfPixelCenters());
            mNpuBackend->setOutputOps(mOp, {interp}, outputs);
        }
        return NO_ERROR;
    }

    auto opName = mOp->name()->str();
    auto param = mOp->main_as_Interp();
    auto xOp = mNpuBackend->getInputOps(mOp);
    auto resizeType = param->resizeType();
    bool alignCorners = param->alignCorners();
    bool halfPixelCenters = param->halfPixelCenters();

    // Newer MNN models store the ONNX coordinate transformation mode in ctm.
    // The legacy flags alone are not sufficient: PyTorch half-pixel models
    // commonly leave halfPixelCenters=false in the FlatBuffer.
    switch (param->ctm()) {
        case CoordinateTransformationMode_NotSet:
            break;
        case CoordinateTransformationMode_AlignCorners:
            alignCorners = true;
            halfPixelCenters = false;
            break;
        case CoordinateTransformationMode_HalfPixels:
            alignCorners = false;
            halfPixelCenters = true;
            break;
        case CoordinateTransformationMode_PytorchHalfPixels:
            if (outputs[0]->height() <= 1 || outputs[0]->width() <= 1) {
                MNN_ERROR("HiAI V320 cannot represent PyTorch half-pixel resize with unit output size\n");
                return NOT_SUPPORT;
            }
            alignCorners = false;
            halfPixelCenters = true;
            break;
        case CoordinateTransformationMode_Asymmetric:
            alignCorners = false;
            halfPixelCenters = false;
            break;
        default:
            MNN_ERROR("HiAI V320 does not support coordinate transformation mode: %d\n", param->ctm());
            return NOT_SUPPORT;
    }
    MNN_ASSERT(resizeType <= 3);
    if (resizeType > 3) {
        MNN_ERROR("npu Interp not support type: %d", resizeType);
        return NOT_SUPPORT;
    }
    vector<int32_t> hw = {outputs[0]->height(),outputs[0]->width()};
    mConstShape = hiai::op::Const(opName + "_w_const");
    {
        ge::TensorDesc fdesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32); 
        ge::TensorPtr filter = std::make_shared<ge::Tensor>();
        filter->SetTensorDesc(fdesc);
        filter->SetData((uint8_t *)hw.data(), hw.size() * sizeof(int32_t));
        mConstShape.set_attr_value(filter);
    }

    // Some HiAI compilers cannot place half-pixel bilinear downsampling on
    // NPUCL.  In particular, Kirin 9020 HCL 100.600 canonicalizes
    // ResizeBilinearV2 to ResizeBilinear and rejects it during BuildV2.  Use
    // the exact depthwise-convolution equivalent for integral downsampling.
    // Keep the verified native V320 ROM600 nodes where that path supports
    // them, because they avoid the larger depthwise kernels.
    const int inputHeight = inputs[0]->height();
    const int inputWidth = inputs[0]->width();
    const int outputHeight = outputs[0]->height();
    const int outputWidth = outputs[0]->width();
    if (inputHeight == outputHeight && inputWidth == outputWidth) {
        mNpuBackend->setOutputOps(mOp, {xOp}, outputs);
        return NO_ERROR;
    }
    const bool isCurrentModelRom600Resize =
        !mNpuBackend->usesHclV600Runtime() &&
        mNpuBackend->supportsRom600NativeResize() && inputs[0]->channel() == 3 &&
        inputHeight == 256 && inputWidth == 256 && outputHeight == outputWidth &&
        (outputHeight == 16 || outputHeight == 32 || outputHeight == 64 ||
         outputHeight == 128);
    if (!isCurrentModelRom600Resize &&
        resizeType == 2 && halfPixelCenters && outputHeight > 1 && outputWidth > 1 &&
        inputHeight > outputHeight && inputWidth > outputWidth &&
        inputHeight % outputHeight == 0 && inputWidth % outputWidth == 0) {
        const int strideY = inputHeight / outputHeight;
        const int strideX = inputWidth / outputWidth;
        if ((strideY % 2) == 0 && (strideX % 2) == 0) {
            const int startY = strideY / 2 - 1;
            const int startX = strideX / 2 - 1;
            const int channels = inputs[0]->channel();
            if (channels != outputs[0]->channel()) {
                return NOT_SUPPORT;
            }
            constexpr size_t kMaxDownsampleWeightElements = 4U * 1024U * 1024U;
            const size_t kernelElements = static_cast<size_t>(strideY) *
                                          static_cast<size_t>(strideX);
            if (channels <= 0 || kernelElements == 0 ||
                kernelElements > kMaxDownsampleWeightElements ||
                static_cast<size_t>(channels) >
                    kMaxDownsampleWeightElements / kernelElements) {
                MNN_ERROR("HiAI half-pixel downsample kernel is too large: %s\n", opName.c_str());
                return NOT_SUPPORT;
            }
            const size_t weightElements = static_cast<size_t>(channels) * kernelElements;
            std::vector<float> weights(weightElements, 0.0f);
            for (int channel = 0; channel < channels; ++channel) {
                const size_t base = static_cast<size_t>(channel) * kernelElements;
                weights[base + startY * strideX + startX] = 0.25f;
                weights[base + startY * strideX + startX + 1] = 0.25f;
                weights[base + (startY + 1) * strideX + startX] = 0.25f;
                weights[base + (startY + 1) * strideX + startX + 1] = 0.25f;
            }
            std::vector<float> bias(channels, 0.0f);

            auto weightConst = std::make_shared<hiai::op::Const>(opName + "_downsample_weight");
            ge::TensorDesc weightDesc(ge::Shape({channels, 1, strideY, strideX}),
                                      ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr weightTensor = std::make_shared<ge::Tensor>();
            weightTensor->SetTensorDesc(weightDesc);
            weightTensor->SetData(reinterpret_cast<const uint8_t*>(weights.data()),
                                  weights.size() * sizeof(float));
            weightConst->set_attr_value(weightTensor);

            auto biasConst = std::make_shared<hiai::op::Const>(opName + "_downsample_bias");
            ge::TensorDesc biasDesc(ge::Shape({channels}), ge::FORMAT_NCHW, ge::DT_FLOAT);
            ge::TensorPtr biasTensor = std::make_shared<ge::Tensor>();
            biasTensor->SetTensorDesc(biasDesc);
            biasTensor->SetData(reinterpret_cast<const uint8_t*>(bias.data()),
                                bias.size() * sizeof(float));
            biasConst->set_attr_value(biasTensor);

            auto depthwise = std::make_shared<hiai::op::ConvolutionDepthwise>(
                opName + "_downsample_depthwise");
            depthwise->set_input_x(*xOp)
                     .set_input_filter(*weightConst)
                     .set_input_bias(*biasConst)
                     .set_attr_strides(ge::AttrValue::LIST_INT({strideY, strideX}))
                     .set_attr_dilations(ge::AttrValue::LIST_INT({1, 1}))
                     .set_attr_pads(ge::AttrValue::LIST_INT({0, 0, 0, 0}))
                     .set_attr_pad_mode("VALID");

            std::vector<std::shared_ptr<ge::Operator>> graphOps = {
                weightConst, biasConst, depthwise};
            mNpuBackend->setOutputOps(mOp, std::move(graphOps), outputs);
            return NO_ERROR;
        }
    }

    if (resizeType == 1) {
        if (halfPixelCenters) {
            MNN_ERROR("HiAI V320 ResizeNearestNeighbor does not support half_pixel_centers\n");
            return NOT_SUPPORT;
        }
        shared_ptr<hiai::op::ResizeNearestNeighbor> interp(new hiai::op::ResizeNearestNeighbor(opName));
        (*interp).set_input_x(*xOp)
                 .set_input_size(mConstShape)
                 .set_attr_align_corners(alignCorners);
        mNpuBackend->setOutputOps(mOp, {interp}, outputs);
    } else if (resizeType == 2) {
        shared_ptr<hiai::op::ResizeBilinearV2> interp(new hiai::op::ResizeBilinearV2(opName));
        (*interp).set_input_x(*xOp)
                 .set_input_size(mConstShape)
                 .set_attr_align_corners(alignCorners)
                 .set_attr_half_pixel_centers(halfPixelCenters);
        mNpuBackend->setOutputOps(mOp, {interp}, outputs);
    } else if (resizeType == 3) {
        MNN_ERROR("HiAI V320 does not expose ResizeBicubic\n");
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

NPUCreatorRegister<TypedCreator<NPUInterp>> __interp_op(OpType_Interp);

} // namespace MNN
