//
//  CPUStridedSlice.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUStridedSlice.hpp"
#include "CPUBackend.hpp"

namespace MNN {

CPUStridedSlice::CPUStridedSlice(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    mDataType = mOp->main_as_StridedSliceParam()->T();
}

ErrorCode CPUStridedSlice::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(4 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    Tensor *input            = inputs[0];
    const int inputDimension = input->buffer().dimensions;
    MNN_ASSERT(inputDimension > 0);

    // input haven't realized
    auto output    = outputs[0];
    auto parameter = mOp->main_as_StridedSliceParam();

    Tensor *begin   = inputs[1];
    Tensor *end     = inputs[2];
    Tensor *strided = inputs[3];

    MNN_ASSERT(begin->buffer().dimensions == end->buffer().dimensions &&
               begin->buffer().dimensions == strided->buffer().dimensions);

    std::vector<int32_t> inputShape(input->buffer().dimensions);
    for (int i = 0; i < input->buffer().dimensions; i++) {
        inputShape[i] = input->buffer().dim[i].extent;
    }

    int stridedSliceDimension = begin->buffer().dim[0].extent;

    std::vector<int32_t> beginShape(stridedSliceDimension);
    std::vector<int32_t> endShape(stridedSliceDimension);
    std::vector<int32_t> stridedShape(stridedSliceDimension);
    std::vector<int32_t> outputShape;
    std::vector<int32_t> outputShapeShrinked;

    std::vector<int32_t> beginMask(stridedSliceDimension);
    for (int i = 0; i < stridedSliceDimension; i++) {
        beginMask[i] = parameter->beginMask() & (1 << i);
    }

    std::vector<int32_t> endMask(stridedSliceDimension);
    for (int i = 0; i < stridedSliceDimension; i++) {
        endMask[i] = parameter->endMask() & (1 << i);
    }

    std::vector<int32_t> shrinkAxisMask(stridedSliceDimension);
    for (int i = 0; i < stridedSliceDimension; i++) {
        shrinkAxisMask[i] = parameter->shrinkAxisMask() & (1 << i);
    }

    int ellipsisMaskNonZeroBitPosition = 0;
    for (int i = 0; i < stridedSliceDimension; i++) {
        int temp = parameter->ellipsisMask() & (1 << i);
        if (temp != 0) {
            ellipsisMaskNonZeroBitPosition = i; // only one non-zero bit is allowed in ellipsisMask
            break;
        }
    }

    std::vector<int32_t> newAxisMask(stridedSliceDimension);
    for (int i = 0; i < stridedSliceDimension; i++) {
        newAxisMask[i] = parameter->newAxisMask() & (1 << i);
    }

    if (parameter->ellipsisMask() != 0 || parameter->newAxisMask() != 0) {
        MNN_ASSERT(false); // TODO: do not support these two mask now
    }

    for (int i = 0; i < stridedSliceDimension; i++) {
        if (beginMask[i] > 0) {
            beginShape[i] = 0;
        } else {
            beginShape[i] = std::min(inputShape[i], begin->host<int32_t>()[i]);
        }
        if (beginShape[i] < 0) {
            beginShape[i] += input->buffer().dim[i].extent;
        }
        assert(beginShape[i] >= 0);
        endShape[i] = endMask[i] > 0
                          ? inputShape[i]
                          : (end->host<int32_t>()[i] > inputShape[i] ? inputShape[i] : end->host<int32_t>()[i]);
        if (endShape[i] < 0) {
            endShape[i] += input->buffer().dim[i].extent;
        }
        assert(endShape[i] >= 0);
        stridedShape[i] = shrinkAxisMask[i] > 0 ? 1 : strided->host<int32_t>()[i];

        if (shrinkAxisMask[i] == 0) {
            int size = (abs(endShape[i] - beginShape[i]) - 1) / abs(stridedShape[i]) + 1;
            outputShape.push_back(size);
            outputShapeShrinked.push_back(size);
        } else {
            outputShape.push_back(1);
        }
    }

    int outputDimensionsWithoutRemain = (int)outputShape.size();
    int dimensionRemained             = input->buffer().dimensions - stridedSliceDimension;

    for (int i = 0; i < dimensionRemained; i++) {
        outputShape.push_back(input->buffer().dim[outputDimensionsWithoutRemain + i].extent);
        outputShapeShrinked.push_back(input->buffer().dim[outputDimensionsWithoutRemain + i].extent);
        stridedShape.push_back(1);
        beginShape.push_back(0);
    }

    output->buffer().dimensions    = (int)outputShapeShrinked.size();
    output->buffer().dim[0].extent = 1;

    for (int i = 0; i < outputShapeShrinked.size(); i++) {
        output->buffer().dim[i].extent = outputShapeShrinked[i];
        output->buffer().dim[i].flags  = 0;
    }

    mBeginShape.clear();
    mEndShape.clear();
    mStrideShape.clear();
    mOutputShape.clear();
    mBeginShape  = beginShape;
    mEndShape    = endShape;
    mStrideShape = stridedShape;
    mOutputShape = outputShape;
    return NO_ERROR;
}

ErrorCode CPUStridedSlice::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input = inputs[0];
    auto output   = outputs[0];
    switch (mDataType) {
        case DataType_DT_INT64:
        case DataType_DT_INT32:
            return execute<int32_t>(input, output);
        case DataType_DT_FLOAT:
        case DataType_DT_DOUBLE:
            return execute<float>(input, output);
        default:
            break;
    }
    return NOT_SUPPORT;
}

template <typename type>
ErrorCode CPUStridedSlice::execute(Tensor *input, Tensor *output) {
    int inputRank = input->buffer().dimensions;
    auto inputData  = input->host<type>();
    auto outputData = output->host<type>();
    if (inputRank == 1) {
        for (int i0 = 0; i0 < mOutputShape[0]; i0++) {
            int dstIndex         = i0;
            int srci0            = mBeginShape[0] + i0 * mStrideShape[0];
            int srcIndex         = srci0;
            outputData[dstIndex] = inputData[srcIndex];
        }
    } else if (inputRank == 2) {
        for (int i0 = 0; i0 < mOutputShape[0]; i0++) {
            for (int i1 = 0; i1 < mOutputShape[1]; i1++) {
                int dstIndex         = i0 * mOutputShape[1] + i1;
                int srci0            = mBeginShape[0] + i0 * mStrideShape[0];
                int srci1            = mBeginShape[1] + i1 * mStrideShape[1];
                int srcIndex         = srci0 * input->buffer().dim[1].extent + srci1;
                outputData[dstIndex] = inputData[srcIndex];
            }
        }
    } else if (inputRank == 3) {
        for (int i0 = 0; i0 < mOutputShape[0]; i0++) {
            for (int i1 = 0; i1 < mOutputShape[1]; i1++) {
                for (int i2 = 0; i2 < mOutputShape[2]; i2++) {
                    int dstIndex = i0 * mOutputShape[1] * mOutputShape[2] + i1 * mOutputShape[2] + i2;
                    int srci0    = mBeginShape[0] + i0 * mStrideShape[0];
                    int srci1    = mBeginShape[1] + i1 * mStrideShape[1];
                    int srci2    = mBeginShape[2] + i2 * mStrideShape[2];
                    int srcIndex = srci0 * input->buffer().dim[1].extent * input->buffer().dim[2].extent +
                                   srci1 * input->buffer().dim[2].extent + srci2;
                    outputData[dstIndex] = inputData[srcIndex];
                }
            }
        }
    } else if (inputRank == 4) {
        for (int i0 = 0; i0 < mOutputShape[0]; i0++) {
            for (int i1 = 0; i1 < mOutputShape[1]; i1++) {
                for (int i2 = 0; i2 < mOutputShape[2]; i2++) {
                    for (int i3 = 0; i3 < mOutputShape[3]; i3++) {
                        int dstIndex = i0 * mOutputShape[1] * mOutputShape[2] * mOutputShape[3] +
                                       i1 * mOutputShape[2] * mOutputShape[3] + i2 * mOutputShape[3] + i3;
                        int srci0    = mBeginShape[0] + i0 * mStrideShape[0];
                        int srci1    = mBeginShape[1] + i1 * mStrideShape[1];
                        int srci2    = mBeginShape[2] + i2 * mStrideShape[2];
                        int srci3    = mBeginShape[3] + i3 * mStrideShape[3];
                        int srcIndex = srci0 * input->buffer().dim[1].extent * input->buffer().dim[2].extent *
                                           input->buffer().dim[3].extent +
                                       srci1 * input->buffer().dim[2].extent * input->buffer().dim[3].extent +
                                       srci2 * input->buffer().dim[3].extent + srci3;
                        outputData[dstIndex] = inputData[srcIndex];
                    }
                }
            }
        }
    }

    return NO_ERROR;
}

class CPUStridedSliceCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUStridedSlice(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUStridedSliceCreator, OpType_StridedSlice);
} // namespace MNN
