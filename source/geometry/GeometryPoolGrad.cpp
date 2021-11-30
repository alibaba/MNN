//
//  GeometryPoolGrad.cpp
//  MNN
//
//  Created by MNN on 2020/06/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvertUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
class GeometryPoolGrad : public GeometryComputer {
public:
    // PoolGrad PoolType_MAXPOOL
    bool onComputeMaxPool(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                          Context& context, CommandBuffer& res) const {
        auto origin       = inputs[0];
        auto originOutput = inputs[1];
        auto inputDiff    = inputs[2];

        auto ow = inputDiff->width();
        auto oh = inputDiff->height();
        auto iw = origin->width();
        auto ih = origin->height();
        auto oc = inputDiff->channel();
        auto ob = inputDiff->batch();

        auto parameter = op->main_as_Pool();
        auto stride_w  = parameter->strideX();
        auto stride_h  = parameter->strideY();
        auto kernel_w  = parameter->kernelX();
        auto kernel_h  = parameter->kernelY();
        auto isGlobal  = parameter->isGlobal();
        auto pad_w     = parameter->padX();
        auto pad_h     = parameter->padY();

        // edit const if global
        if (isGlobal) {
            kernel_w = iw;
            kernel_h = ih;
            stride_w = iw;
            stride_h = ih;
            pad_w    = 0;
            pad_h    = 0;
        } else {
            if (parameter->padType() == PoolPadType_SAME) {
                int pad_w_total = (ow - 1) * stride_w + kernel_w - iw;
                int pad_h_total = (oh - 1) * stride_h + kernel_h - ih;
                pad_w           = pad_w_total > 0 ? pad_w_total / 2 : 0;
                pad_h           = pad_h_total > 0 ? pad_h_total / 2 : 0;
            } else if (parameter->padType() == PoolPadType_VALID) {
                pad_w = 0;
                pad_h = 0;
            }
        }

        std::vector<std::shared_ptr<Tensor>> originSplit;
        originSplit.resize(kernel_h * kernel_w);

        std::vector<std::shared_ptr<Tensor>> originGEqual;
        originGEqual.resize(kernel_h * kernel_w);

        std::vector<std::shared_ptr<Tensor>> originDiff;
        originDiff.resize(kernel_h * kernel_w);

        std::vector<std::shared_ptr<Tensor>> inpDiffAdd;
        inpDiffAdd.resize(kernel_h * kernel_w);

        for (int ky = 0; ky < kernel_h; ky++) {
            auto startSy = ky - pad_h;
            int startDy  = 0;
            if (startSy < 0) {
                startDy = ((-startSy) + stride_h - 1) / stride_h;
                startSy = startSy + startDy * stride_h;
            }
            auto endDy = oh - 1;
            auto endSy = endDy * stride_h + ky - pad_h;
            if (endSy >= ih) {
                endDy = endDy - (endSy - ih + stride_h) / stride_h;
                endSy = endDy * stride_h + ky - pad_h;
            }
            if (startDy > endDy) {
                continue;
            }
            MNN_ASSERT(endDy >= 0);
            MNN_ASSERT(startDy < oh);

            for (int kx = 0; kx < kernel_w; kx++) {
                auto startSx = kx - pad_w;
                int startDx  = 0;
                if (startSx < 0) {
                    startDx = ((-startSx) + stride_w - 1) / stride_w;
                    startSx = startSx + startDx * stride_w;
                }
                auto endDx = ow - 1;
                auto endSx = endDx * stride_w + kx - pad_w;
                if (endSx >= iw) {
                    endDx = endDx - (endSx - iw + stride_w) / stride_w;
                    endSx = endDx * stride_w + kx - pad_w;
                }
                if (startDx > endDx) {
                    continue;
                }
                MNN_ASSERT(endDx >= 0);
                MNN_ASSERT(startDx < ow);

                // A: Input feature
                int index = ky * kernel_w + kx;
                originSplit[index].reset(new Tensor);
                originSplit[index]->buffer().type       = halide_type_of<float>();
                originSplit[index]->buffer().dimensions = 4;
                originSplit[index]->setLength(0, ob);
                originSplit[index]->setLength(1, oc);
                originSplit[index]->setLength(2, oh);
                originSplit[index]->setLength(3, ow);
                auto des             = TensorUtils::getDescribe(originSplit[index].get());
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

                Tensor::InsideDescribe::Region region;
                region.origin        = origin;
                region.size[0]       = ob * oc;
                region.size[1]       = endDy - startDy + 1;
                region.size[2]       = endDx - startDx + 1;
                region.dst.offset    = startDy * ow + startDx;
                region.src.offset    = startSy * iw + startSx;
                region.dst.stride[0] = ow * oh;
                region.src.stride[0] = iw * ih;
                region.dst.stride[1] = ow;
                region.src.stride[1] = iw * stride_h;
                region.dst.stride[2] = 1;
                region.src.stride[2] = stride_w;
                des->regions.emplace_back(std::move(region));

                // greater equal
                std::shared_ptr<Tensor> originGEqualInt(new Tensor);
                originGEqualInt->buffer().type       = halide_type_of<int32_t>();
                originGEqualInt->buffer().dimensions = 4;
                originGEqualInt->setLength(0, ob);
                originGEqualInt->setLength(1, oc);
                originGEqualInt->setLength(2, oh);
                originGEqualInt->setLength(3, ow);
                des = TensorUtils::getDescribe(originGEqualInt.get());
                // des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

                auto cmd = GeometryComputerUtils::makeBinary(BinaryOpOperation_GREATER_EQUAL, originSplit[index].get(),
                                                             originOutput, originGEqualInt.get());

                // cast int to float
                originGEqual[index].reset(new Tensor);
                originGEqual[index]->buffer().type       = halide_type_of<float>();
                originGEqual[index]->buffer().dimensions = 4;
                originGEqual[index]->setLength(0, ob);
                originGEqual[index]->setLength(1, oc);
                originGEqual[index]->setLength(2, oh);
                originGEqual[index]->setLength(3, ow);
                des = TensorUtils::getDescribe(originGEqual[index].get());
                // des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

                std::unique_ptr<OpT> cast2float(new OpT);
                cast2float->type                     = OpType_Cast;
                cast2float->main.type                = OpParameter_CastParam;
                cast2float->main.value               = new CastParamT;
                cast2float->main.AsCastParam()->dstT = DataType_DT_FLOAT;

                flatbuffers::FlatBufferBuilder builder1;
                auto lastOffset1 = Op::Pack(builder1, cast2float.get());
                builder1.Finish(lastOffset1);
                auto cmd1 = GeometryComputerUtils::makeCommand(builder1, {originGEqualInt.get()}, {originGEqual[index].get()});
                // mul inputDiff
                originDiff[index].reset(new Tensor);
                originDiff[index]->buffer().type       = halide_type_of<float>();
                originDiff[index]->buffer().dimensions = 4;
                originDiff[index]->setLength(0, ob);
                originDiff[index]->setLength(1, oc);
                originDiff[index]->setLength(2, oh);
                originDiff[index]->setLength(3, ow);
                des = TensorUtils::getDescribe(originDiff[index].get());
                // des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

                auto cmd2 = GeometryComputerUtils::makeBinary(BinaryOpOperation_MUL, inputDiff,
                                                              originGEqual[index].get(), originDiff[index].get());

                // expand tensor
                inpDiffAdd[index].reset(new Tensor);
                inpDiffAdd[index]->buffer().type       = halide_type_of<float>();
                inpDiffAdd[index]->buffer().dimensions = 4;
                inpDiffAdd[index]->setLength(0, ob);
                inpDiffAdd[index]->setLength(1, oc);
                inpDiffAdd[index]->setLength(2, ih);
                inpDiffAdd[index]->setLength(3, iw);
                des                  = TensorUtils::getDescribe(inpDiffAdd[index].get());
                des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
                des->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;

                // Tensor::InsideDescribe::Region region;
                region.origin        = originDiff[index].get();
                region.size[0]       = ob * oc;
                region.size[1]       = oh;
                region.size[2]       = ow;
                region.src.offset    = 0;
                region.dst.offset    = ky * iw + kx;
                region.src.stride[0] = ow * oh;
                region.dst.stride[0] = iw * ih;
                region.src.stride[1] = ow;
                region.dst.stride[1] = iw * stride_h;
                region.src.stride[2] = 1;
                region.dst.stride[2] = stride_w;
                des->regions.emplace_back(std::move(region));

                res.extras.emplace_back(inpDiffAdd[index]);
                res.extras.emplace_back(originSplit[index]);
                res.extras.emplace_back(originGEqual[index]);
                res.extras.emplace_back(originGEqualInt);
                res.extras.emplace_back(originDiff[index]);
                res.command.emplace_back(std::move(cmd));
                res.command.emplace_back(std::move(cmd1));
                res.command.emplace_back(std::move(cmd2));
            }
        }

        // eltwise
        std::shared_ptr<Tensor> tmpOutput(new Tensor);
        {
            std::unique_ptr<OpT> eltWise(new OpT);
            eltWise->type                   = OpType_Eltwise;
            eltWise->main.type              = OpParameter_Eltwise;
            eltWise->main.value             = new EltwiseT;
            eltWise->main.AsEltwise()->type = EltwiseType_SUM;
            // eltWise->main.AsEltwise()->coeff() = nullptr;
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, eltWise.get());
            builder.Finish(lastOffset);
            std::vector<Tensor*> inputs(kernel_w * kernel_h);
            for (int i = 0; i < kernel_w * kernel_h; i++) {
                inputs[i] = inpDiffAdd[i].get();
            }
            auto cmd = GeometryComputerUtils::makeCommand(builder, inputs, outputs);
            res.command.emplace_back(std::move(cmd));
        }
        return true;
    }

    // PoolGrad PoolType_AVEPOOL
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        auto parameter = op->main_as_Pool();
        if (parameter->type() == PoolType_MAXPOOL) {
            return onComputeMaxPool(op, inputs, outputs, context, res);
        } else if (parameter->type() != PoolType_AVEPOOL) {
            MNN_PRINT("Pool type not supported!\n");
            return false;
        }

        auto origin     = inputs[0];
        auto inputDiff  = inputs[2];
        auto outputDiff = outputs[0];

        auto ow = inputDiff->width();
        auto oh = inputDiff->height();
        auto iw = origin->width();
        auto ih = origin->height();
        auto oc = inputDiff->channel();
        auto ob = inputDiff->batch();

        auto stride_w = parameter->strideX();
        auto stride_h = parameter->strideY();
        auto kernel_w = parameter->kernelX();
        auto kernel_h = parameter->kernelY();
        auto isGlobal = parameter->isGlobal();
        auto pad_w    = parameter->padX();
        auto pad_h    = parameter->padY();

        // edit const if global
        if (isGlobal) {
            kernel_w = iw;
            kernel_h = ih;
            stride_w = iw;
            stride_h = ih;
            pad_w    = 0;
            pad_h    = 0;
        } else {
            if (parameter->padType() == PoolPadType_SAME) {
                int pad_w_total = (ow - 1) * stride_w + kernel_w - iw;
                int pad_h_total = (oh - 1) * stride_h + kernel_h - ih;
                pad_w           = pad_w_total > 0 ? pad_w_total / 2 : 0;
                pad_h           = pad_h_total > 0 ? pad_h_total / 2 : 0;
            } else if (parameter->padType() == PoolPadType_VALID) {
                pad_w = 0;
                pad_h = 0;
            }
        }
        std::shared_ptr<Tensor> inpDifTrans;

        inpDifTrans.reset(new Tensor);
        inpDifTrans->buffer().type       = halide_type_of<float>();
        inpDifTrans->buffer().dimensions = 5;
        inpDifTrans->setLength(0, kernel_h * kernel_w);
        inpDifTrans->setLength(1, ob);
        inpDifTrans->setLength(2, oc);
        inpDifTrans->setLength(3, ih);
        inpDifTrans->setLength(4, iw);
        auto des             = TensorUtils::getDescribe(inpDifTrans.get());
        des->memoryType      = Tensor::InsideDescribe::MEMORY_VIRTUAL;
        des->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        des->regions.clear();
        // des->regions.reserve(kernel_h*kernel_w);

        for (int ky = 0; ky < kernel_h; ky++) {
            auto startSy = ky - pad_h;
            int startDy  = 0;
            if (startSy < 0) {
                startDy = ((-startSy) + stride_h - 1) / stride_h;
                startSy = startSy + startDy * stride_h;
            }
            auto endDy = oh - 1;
            auto endSy = endDy * stride_h + ky - pad_h;
            if (endSy >= ih) {
                endDy = endDy - (endSy - ih + stride_h) / stride_h;
                endSy = endDy * stride_h + ky - pad_h;
            }
            if (startDy > endDy) {
                continue;
            }
            MNN_ASSERT(endDy >= 0);
            MNN_ASSERT(startDy < oh);

            for (int kx = 0; kx < kernel_w; kx++) {
                auto startSx = kx - pad_w;
                int startDx  = 0;
                if (startSx < 0) {
                    startDx = ((-startSx) + stride_w - 1) / stride_w;
                    startSx = startSx + startDx * stride_w;
                }
                auto endDx = ow - 1;
                auto endSx = endDx * stride_w + kx - pad_w;
                if (endSx >= iw) {
                    endDx = endDx - (endSx - iw + stride_w) / stride_w;
                    endSx = endDx * stride_w + kx - pad_w;
                }
                if (startDx > endDx) {
                    continue;
                }
                MNN_ASSERT(endDx >= 0);
                MNN_ASSERT(startDx < ow);

                // A: Input feature
                int index = ky * kernel_w + kx;

                Tensor::InsideDescribe::Region region;
                region.origin        = inputDiff;
                region.size[0]       = ob * oc;
                region.size[1]       = endDy - startDy + 1;
                region.size[2]       = endDx - startDx + 1;
                region.src.offset    = startDy * ow + startDx;
                region.dst.offset    = index * ob * oc * ih * iw + startSy * iw + startSx;
                region.src.stride[0] = ow * oh;
                region.dst.stride[0] = iw * ih;
                region.src.stride[1] = ow;
                region.dst.stride[1] = iw * stride_h;
                region.src.stride[2] = 1;
                region.dst.stride[2] = stride_w;
                des->regions.emplace_back(std::move(region));
            }
        }
        res.extras.emplace_back(inpDifTrans);

        // reduction mean
        std::shared_ptr<Tensor> tmpOutput;
        {
            tmpOutput.reset(new Tensor);
            tmpOutput->buffer().type       = halide_type_of<float>();
            tmpOutput->buffer().dimensions = 4;
            tmpOutput->setLength(0, ob);
            tmpOutput->setLength(1, oc);
            tmpOutput->setLength(2, ih);
            tmpOutput->setLength(3, iw);
            auto des = TensorUtils::getDescribe(tmpOutput.get());
            // des->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            des->dimensionFormat = MNN_DATA_FORMAT_NCHW;

            std::unique_ptr<OpT> mean(new OpT);
            mean->type                               = OpType_Reduction;
            mean->main.type                          = OpParameter_ReductionParam;
            mean->main.value                         = new ReductionParamT;
            mean->main.AsReductionParam()->dim       = {0};
            mean->main.AsReductionParam()->keepDims  = false;
            mean->main.AsReductionParam()->operation = ReductionType_MEAN;

            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, mean.get());
            builder.Finish(lastOffset);
            auto cmd = GeometryComputerUtils::makeCommand(builder, {inpDifTrans.get()}, {tmpOutput.get()});
            auto outputDes        = TensorUtils::getDescribe(outputs[0]);
            outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            Tensor::InsideDescribe::Region desReg;
            desReg.size[0]       = ob * oc;
            desReg.size[1]       = ih;
            desReg.size[2]       = iw;
            desReg.dst.offset    = 0;
            desReg.dst.stride[0] = ih * iw;
            desReg.dst.stride[1] = iw;
            desReg.dst.stride[2] = 1;
            desReg.src.offset    = 0;
            desReg.src.stride[0] = ih * iw;
            desReg.src.stride[1] = iw;
            desReg.src.stride[2] = 1;
            desReg.origin        = tmpOutput.get();
            outputDes->regions.emplace_back(std::move(desReg));

            res.extras.emplace_back(std::move(tmpOutput));
            res.command.emplace_back(std::move(cmd));
        }

        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryPoolGrad);
    GeometryComputer::registerGeometryComputer(comp, {OpType_PoolGrad});
}

REGISTER_GEOMETRY(GeometryPoolGrad, _create);

} // namespace MNN
