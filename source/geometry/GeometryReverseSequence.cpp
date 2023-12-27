//
//  GeometryReverseSequence.cpp
//  MNN
//
//  Created by MNN on 2020/05/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "geometry/GeometryComputer.hpp"
namespace MNN {
class GeometryReverseSequence : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(2 == inputs.size());
        auto output  = outputs[0];
        auto input   = inputs[0];
        auto reverse = inputs[1];
        // below will using stride, set it first
        TensorUtils::setLinearLayout(output);
        TensorUtils::setLinearLayout(input);
        TensorUtils::setLinearLayout(reverse);

        if (nullptr == op->main_as_ReverseSequenceParam()) {
            MNN_ERROR("Dont's has Parameters for OpType_ReverseSequence\n");
            return false;
        }
        auto seqDim = op->main_as_ReverseSequenceParam()->seqDim();
        if (seqDim < 0) {
            seqDim += inputs[0]->dimensions();
        }
        auto batchDim = op->main_as_ReverseSequenceParam()->batchDim();
        if (batchDim < 0) {
            batchDim += inputs[0]->dimensions();
        }
        if (seqDim == batchDim) {
            MNN_ERROR("seq and batch dim can't be the same\n");
            return false;
        }
        if (inputs[0]->getType().bits != 32) {
            MNN_ERROR("Don't support %d bit's ReverseSequence\n", inputs[0]->getType().bits);
            return false;
        }

        if (inputs[1]->length(0) != inputs[0]->length(batchDim)) {
            MNN_ERROR("ReverseSequence info error\n");
            return false;
        }

        int mid0 = seqDim;
        int mid1 = batchDim;
        if (mid0 > mid1) {
            auto temp = mid1;
            mid1      = mid0;
            mid0      = temp;
        }
        int mInsideStride = inputs[0]->stride(mid1);

        int mOutsideSize   = 1;
        int mOutSideStride = 1;
        for (int i = 0; i < mid0; ++i) {
            mOutsideSize *= inputs[0]->length(i);
        }
        if (mid0 > 0) {
            mOutSideStride = inputs[0]->stride(mid0 - 1);
        }

        int mMidSize   = 1;
        int mMidStride = 1;
        for (int i = mid0 + 1; i < mid1; ++i) {
            mMidSize *= inputs[0]->length(i);
        }
        if (mid1 > 0) {
            mMidStride = inputs[0]->stride(mid1 - 1);
        }

        auto outputDes        = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        auto batchSize = input->length(batchDim);
        outputDes->regions.clear();

        for (int batch = 0; batch < batchSize; ++batch) {
            auto q = reverse->host<int32_t>()[batch];
            if (q > input->length(seqDim) || q < 1) {
                MNN_ERROR("ReverseSequence info error\n");
                return false;
            }

            for (int o = 0; o < mOutsideSize; ++o) {
                Tensor::InsideDescribe::Region dstSlice;
                dstSlice.origin = input;

                dstSlice.size[0] = q;
                dstSlice.size[1] = mMidSize;
                dstSlice.size[2] = mInsideStride;

                dstSlice.src.stride[0] = -(input->stride(seqDim));
                dstSlice.src.stride[1] = mMidStride;
                dstSlice.src.stride[2] = 1;
                dstSlice.src.offset =
                    (q - 1) * input->stride(seqDim) + batch * input->stride(batchDim) + o * mOutSideStride;

                dstSlice.dst.offset    = batch * output->stride(batchDim) + o * mOutSideStride;
                dstSlice.dst.stride[0] = output->stride(seqDim);
                dstSlice.dst.stride[1] = mMidStride;
                dstSlice.dst.stride[2] = 1;

                outputDes->regions.emplace_back(std::move(dstSlice));
            }
            
            if(q < input->length(seqDim)) {
                const int leftSeq = input->length(seqDim) - q;
                for (int o = 0; o < mOutsideSize; ++o) {
                    Tensor::InsideDescribe::Region dstSlice;
                    dstSlice.origin = input;

                    dstSlice.size[0] = leftSeq;
                    dstSlice.size[1] = mMidSize;
                    dstSlice.size[2] = mInsideStride;

                    dstSlice.src.stride[0] = input->stride(seqDim);
                    dstSlice.src.stride[1] = mMidStride;
                    dstSlice.src.stride[2] = 1;
                    dstSlice.src.offset =
                        q * input->stride(seqDim) + batch * input->stride(batchDim) + o * mOutSideStride;

                    dstSlice.dst.offset    = q * output->stride(seqDim) + batch * output->stride(batchDim) + o * mOutSideStride;
                    dstSlice.dst.stride[0] = output->stride(seqDim);
                    dstSlice.dst.stride[1] = mMidStride;
                    dstSlice.dst.stride[2] = 1;

                    outputDes->regions.emplace_back(std::move(dstSlice));
                }
            }
        }
        return true;
    }
};

class GeometryReverse : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
        MNN_ASSERT(1 == outputs.size());
        MNN_ASSERT(2 == inputs.size());
        auto output  = outputs[0];
        auto input   = inputs[0];
        int  axis    = inputs[1]->host<int>()[0];
        int outsideSize = 1, insideSize = 1, reverseSize = input->length(axis);
        for (int i = 0; i < input->dimensions(); i++) {
            if (i < axis) {
                outsideSize *= input->length(i);
            }
            if (i > axis) {
                insideSize *= input->length(i);
            }
        }
        auto outputDes        = TensorUtils::getDescribe(output);
        outputDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

        Tensor::InsideDescribe::Region region;
        region.origin = input;

        region.size[0] = outsideSize;
        region.size[1] = reverseSize;
        region.size[2] = insideSize;

        region.src.offset = reverseSize * insideSize - insideSize;
        region.src.stride[0] = reverseSize*insideSize;
        region.src.stride[1] = -insideSize;
        region.src.stride[2] = 1;

        region.dst.offset    = 0;
        region.dst.stride[0] = reverseSize*insideSize;
        region.dst.stride[1] = insideSize;
        region.dst.stride[2] = 1;
        outputDes->regions.emplace_back(std::move(region));

        return true;
    }
};
static void _create() {
    std::shared_ptr<GeometryComputer> comp(new GeometryReverseSequence);
    GeometryComputer::registerGeometryComputer(comp, {OpType_ReverseSequence});
    std::shared_ptr<GeometryComputer> comp1(new GeometryReverse);
    GeometryComputer::registerGeometryComputer(comp1, {OpType_Reverse});
}

REGISTER_GEOMETRY(GeometryReverseSequence, _create);

} // namespace MNN
