//
//  GeometryDilation2D.cpp
//  MNN
//
//  Created by MNN on 2020/8/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvertUtils.hpp"
#include "GeometryConvUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {

class GeometryDilation2D : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Context& context, CommandBuffer& res) const override {
        auto input  = inputs[0];
        auto output = outputs[0];
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        MNN_ASSERT(TensorUtils::getDescribe(output)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        auto weightData = op->main_as_Convolution2D()->weight();
        auto common     = op->main_as_Convolution2D()->common();
        const int depth = common->outputCount();
        const int inputChannel = input->length(1), batch = input->length(0), outputChannel = output->length(1);
        MNN_ASSERT(depth == inputChannel && depth == outputChannel);
        const int kernelHeight = common->kernelY(), kernelWidth = common->kernelX();
        const int strideHeight = common->strideY(), strideWidth = common->strideX();
        const int dialteHeight = common->dilateY(), dialteWidth = common->dilateX();
        const int outputHeight = output->length(2), outputWidth = output->length(3);
        const int inputHeight = input->length(2), inputWidth = input->length(3);
        auto pads  = ConvolutionCommon::convolutionPad(input, output, common);
        auto weightTensor = context.allocConst(op, {static_cast<int>(weightData->size())}, halide_type_of<float>());
        ::memcpy(weightTensor.get()->host<float>(), weightData->data(), weightData->size()*sizeof(float));
        auto weight = weightTensor.get();

        const int kernelSize = depth * kernelHeight * kernelWidth;
        const int computeNum = batch * outputHeight * outputWidth;
        // compute pipline:
        // A : input  ===im2col===> A [(ic * kh * kw) * (batch * oh * ow)]
        // B : weight ==broadcast=> B [(ic * kh * kw) * (batch * oh * ow)]
        // C : A + B  ============> C [(ic * kh * kw) * (bacth * oh * ow)]
        // D : C      ===reshape==> D [ic * (kh * kw) * (batch * oh * ow)]
        // E : max(D, dim = 1) ===> E [ic * 1 * (batch * oh * ow)]
        // output : E ==transpose=> output [batch * ic * oh * ow]
        Tensor *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr, *E = nullptr;
        {
            // dilation's result value is the max value exclude pad value,
            // set -inf as pad value so it's value wont appear in result
            auto padVal = context.allocConst(op, {1}, halide_type_of<float>());
            padVal->host<float>()[0] = -std::numeric_limits<float>::infinity();
            // Im2Col: n, ic, ih, iw -> (ic * kh * kw) * (batch * oh * ow)
            std::shared_ptr<Tensor> im2Col(new Tensor);
            auto tmpT = GeometryConvUtils::im2Col(im2Col.get(), input, inputChannel, kernelHeight, kernelWidth, batch,
                                      outputHeight, outputWidth, inputHeight, inputWidth, strideHeight,
                                      strideWidth, dialteHeight, dialteWidth, pads, 0, padVal.get());
            if (nullptr != tmpT.get()) {
                res.extras.emplace_back(tmpT);
            }
            A = im2Col.get();
            res.extras.emplace_back(im2Col);
        }
        {
            // broadcast weight => weight * computeNum
            std::shared_ptr<Tensor> kernel(new Tensor);
            B = kernel.get();
            kernel->buffer().type       = halide_type_of<float>();
            kernel->buffer().dimensions = 2;
            kernel->setLength(0, kernelSize);
            kernel->setLength(1, computeNum);
            TensorUtils::setLinearLayout(kernel.get());
            auto des             = TensorUtils::getDescribe(kernel.get());
            des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            des->regions.clear();
            des->regions.reserve(computeNum);
            for (int i = 0; i < computeNum; i++) {
                Tensor::InsideDescribe::Region region;
                region.origin        = weight;
                region.size[2]       = kernelSize;
                region.dst.stride[2] = computeNum;
                region.dst.offset    = i;
                des->regions.emplace_back(std::move(region));
            }
            res.extras.emplace_back(std::move(kernel));
        }
        {
            std::shared_ptr<Tensor> addValue;
            addValue.reset(Tensor::createDevice<float>({kernelSize, computeNum}));
            C = addValue.get();
            auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_ADD, A, B, C);
            res.extras.emplace_back(addValue);
            res.command.emplace_back(std::move(cmd));
        }
        {
            std::shared_ptr<Tensor> addValueReshape(new Tensor);
            D = addValueReshape.get();
            addValueReshape->buffer().type = halide_type_of<float>();
            addValueReshape->buffer().dimensions = 3;
            addValueReshape->setLength(0, depth);
            addValueReshape->setLength(1, kernelHeight*kernelWidth);
            addValueReshape->setLength(2, computeNum);
            TensorUtils::setLinearLayout(D);
            auto kernelDiffDes = TensorUtils::getDescribe(D);
            kernelDiffDes->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            kernelDiffDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto totalSlice = TensorUtils::makeFullSlice(C);
            kernelDiffDes->regions.emplace_back(std::move(totalSlice));
            res.extras.emplace_back(addValueReshape);
        }
        {
            std::shared_ptr<Tensor> maxValue;
            maxValue.reset(Tensor::createDevice<float>({depth, 1, computeNum}, Tensor::CAFFE));
            E = maxValue.get();
            auto cmd = GeometryComputerUtils::makeReduce(ReductionType_MAXIMUM, D, E);
            res.extras.emplace_back(maxValue);
            res.command.emplace_back(std::move(cmd));
        }
        {
            // E [ic * 1 * (batch * oh * ow)] -> output [batch * ic * oh * ow]
            auto des             = TensorUtils::getDescribe(output);
            des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
            des->regions.clear();
            // auto totalSlice = TensorUtils::makeFullSlice(E);
            // des->regions.emplace_back(std::move(totalSlice));
            des->regions.reserve(batch);
            Tensor::InsideDescribe::Region region;
            region.origin        = E;
            region.size[0]       = batch;
            region.size[1]       = depth;
            region.size[2]       = outputHeight * outputWidth;
            region.src.stride[0] = outputHeight * outputWidth;
            region.src.stride[1] = batch * outputHeight * outputWidth;
            region.dst.stride[0] = depth * outputHeight * outputWidth;
            region.dst.stride[1] = outputHeight * outputWidth;
            des->regions.emplace_back(std::move(region));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryDilation2D);
    GeometryComputer::registerGeometryComputer(comp, {OpType_Dilation2D});
}

REGISTER_GEOMETRY(GeometryDilation2D, _create);

} // namespace MNN
