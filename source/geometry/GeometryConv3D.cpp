//
//  GeometryConv3D.cpp
//  MNN
//
//  Created by MNN on 2020/7/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvertUtils.hpp"
#include "GeometryConvUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {

class GeometryConv3D : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        auto input      = inputs[0];
        auto output = outputs[0];
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        MNN_ASSERT(TensorUtils::getDescribe(output)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        auto biasData = op->main_as_Convolution3D()->bias();
        auto weightData = op->main_as_Convolution3D()->weight();
        auto common     = op->main_as_Convolution3D()->common();
        auto kernels = common->kernels();
        auto strides = common->strides();
        auto pads = common->pads();
        auto dialtes = common->dilates();
        const int kernelDepth = kernels->Get(0), kernelHeight = kernels->Get(1), kernelWidth = kernels->Get(2);
        const int strideDepth = strides->Get(0), strideHeight = strides->Get(1), strideWidth = strides->Get(2);
        const int dialteDepth = dialtes->Get(0), dialteHeight = dialtes->Get(1), dialteWidth = dialtes->Get(2);
        const int padDepth = pads->Get(0), padHeight = pads->Get(1), padWidth = pads->Get(2);
        const int outputDepth = output->length(2), outputHeight = output->length(3), outputWidth = output->length(4);
        const int inputDepth = input->length(2), inputHeight = input->length(3), inputWidth = input->length(4);
        const int inputChannel = input->length(1), batch = input->length(0), outputChannel = output->length(1);

        auto weightTensor = context.allocConst(op, {static_cast<int>(weightData->size())}, halide_type_of<float>());
        ::memcpy(weightTensor.get()->host<float>(), weightData->data(), weightData->size()*sizeof(float));
        auto weight = weightTensor.get();
        auto biasTensor = context.allocConst(op, {outputChannel}, halide_type_of<float>());
        ::memcpy(biasTensor.get()->host<float>(), biasData->data(), biasData->size()*sizeof(float));
        auto bias = biasTensor.get();

        Tensor* A = nullptr;
        Tensor* B = nullptr;
        {
            // B: Input Im2Col, n, ic, id, ih, iw -> ic*kd*kh*kw*n*od*oh*ow
            std::shared_ptr<Tensor> im2Col(new Tensor);
            GeometryConvUtils::im2Col3d(im2Col.get(), input, inputChannel, kernelDepth, kernelHeight, kernelWidth,
            batch, outputDepth, outputHeight, outputWidth, inputDepth, inputHeight, inputWidth,
            strideDepth, strideHeight, strideWidth, dialteDepth, dialteHeight, dialteWidth, padDepth, padHeight, padWidth);
            B = im2Col.get();
            res.extras.emplace_back(im2Col);
        }
        {
            // A: Weight oc, ic, kd, kh, kw -> oc, ic*kd*kh*kw
            std::shared_ptr<Tensor> kernel(new Tensor);
            A                           = kernel.get();
            kernel->buffer().type       = halide_type_of<float>();
            kernel->buffer().dimensions = 2;
            kernel->setLength(0, outputChannel);
            kernel->setLength(1, inputChannel*kernelDepth*kernelHeight*kernelWidth);
            auto des             = TensorUtils::getDescribe(kernel.get());
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            GeometryComputerUtils::makeRawAddressRef(kernel.get(), weight, 0, inputChannel*kernelDepth*kernelHeight*kernelWidth * outputChannel);
            res.extras.emplace_back(std::move(kernel));
        }
        {
            // C = MatMul(B, A)
            std::shared_ptr<Tensor> C(new Tensor);
            C->buffer().type       = halide_type_of<float>();
            C->buffer().dimensions = 2;
            C->setLength(0, batch * outputDepth * outputHeight * outputWidth);
            C->setLength(1, outputChannel);
            TensorUtils::getDescribe(C.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            res.command.emplace_back(GeometryComputerUtils::makeMatMul(B, A, C.get(), bias, true, true));
            res.extras.emplace_back(C);
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
                C2->buffer().dimensions = 2;
                C2->setLength(0, batch * outputDepth * outputHeight * outputWidth);
                C2->setLength(1, outputChannel);
                TensorUtils::getDescribe(C2.get())->dimensionFormat = MNN_DATA_FORMAT_NCHW;
                auto cmd = GeometryComputerUtils::makeCommand(builder, {C.get()}, {C2.get()});
                res.command.emplace_back(cmd);
                res.extras.emplace_back(C2);
                C = C2;
            }
            // Transpose
            // Batch, od, oh, ow, oc -> batch, oc, od, oh, ow
            TensorUtils::setLinearLayout(C.get());
            if (outputDepth * outputWidth * outputHeight == 1) {
                GeometryComputerUtils::makeRawAddressRef(outputs[0], C.get(), 0, batch * outputChannel);
            } else {
                auto kernelDiffDes        = TensorUtils::getDescribe(outputs[0]);
                kernelDiffDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                kernelDiffDes->regions.resize(1);
                auto& desReg         = kernelDiffDes->regions[0];
                desReg.size[0]       = batch;
                desReg.size[1]       = outputChannel;
                desReg.size[2]       = outputDepth * outputHeight * outputWidth;
                desReg.dst.offset    = 0;
                desReg.dst.stride[0] = outputChannel * outputDepth * outputHeight * outputWidth;
                desReg.dst.stride[1] = outputDepth * outputHeight * outputWidth;
                desReg.dst.stride[2] = 1;
                desReg.src.offset    = 0;
                desReg.src.stride[0] = outputChannel * outputDepth * outputHeight * outputWidth;
                desReg.src.stride[1] = 1;
                desReg.src.stride[2] = outputChannel;
                desReg.origin        = C.get();
            }
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryConv3D);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Convolution3D});
}

REGISTER_GEOMETRY(GeometryConv3D, _create);

} // namespace MNN
