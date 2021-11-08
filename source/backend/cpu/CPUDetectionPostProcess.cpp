//
//  CPUDetectionPostProcess.cpp
//  MNN
//
//  Created by MNN on 2019/10/29.
//  Copyright © 2018, Alibaba Group Holding Limited

#include <math.h>
#include <numeric>

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUDetectionPostProcess.hpp"
#include "backend/cpu/CPUNonMaxSuppressionV2.hpp"

namespace MNN {

static void _decodeBoxes(const Tensor* boxesEncoding, const Tensor* anchors, const CenterSizeEncoding& scaleValues,
                         Tensor* decodeBoxes) {
    const int numBoxes        = boxesEncoding->length(1);
    const int boxCoordNum     = boxesEncoding->length(2);
    const int numAnchors      = anchors->length(0);
    const int anchorsCoordNum = anchors->length(1);
    MNN_CHECK(numBoxes == numAnchors, "the number of input boxes should be equal to the number of anchors!");
    MNN_CHECK(boxCoordNum >= 4, "input box encoding ERROR!");
    MNN_CHECK(anchorsCoordNum == 4, "input anchors ERROR!");

    const auto boxesPtr   = boxesEncoding->host<float>();
    const auto anchorsPtr = reinterpret_cast<const CenterSizeEncoding*>(anchors->host<float>());
    auto decodeBoxesPtr   = reinterpret_cast<BoxCornerEncoding*>(decodeBoxes->host<float>());

    CenterSizeEncoding boxCenterSize;
    CenterSizeEncoding anchor;
    for (int idx = 0; idx < numBoxes; ++idx) {
        const int boxIndex = idx * boxCoordNum;
        boxCenterSize      = *reinterpret_cast<const CenterSizeEncoding*>(boxesPtr + boxIndex);
        anchor             = anchorsPtr[idx];
        float ycenter      = boxCenterSize.y / scaleValues.y * anchor.h + anchor.y;
        float xcenter      = boxCenterSize.x / scaleValues.x * anchor.w + anchor.x;
        float halfh        = 0.5f * static_cast<float>(exp(boxCenterSize.h / scaleValues.h)) * anchor.h;
        float halfw        = 0.5f * static_cast<float>(exp(boxCenterSize.w / scaleValues.w)) * anchor.w;
        auto& curBox       = decodeBoxesPtr[idx];
        curBox.ymin        = ycenter - halfh;
        curBox.xmin        = xcenter - halfw;
        curBox.ymax        = ycenter + halfh;
        curBox.xmax        = xcenter + halfw;
    }
}

static void _NonMaxSuppressionMultiClassFastImpl(const DetectionPostProcessParamT& postProcessParam,
                                                 const Tensor* decodedBoxes, const Tensor* classPredictions,
                                                 Tensor* detectionBoxes, Tensor* detectionClass,
                                                 Tensor* detectionScores, Tensor* numDetections) {
    // decoded_boxes shape is [numBoxes, 4]
    const int numBoxes               = decodedBoxes->length(0);
    const int numClasses             = postProcessParam.numClasses;
    const int maxClassesPerAnchor    = postProcessParam.maxClassesPerDetection;
    const int numClassWithBackground = classPredictions->length(2);

    const int labelOffset = numClassWithBackground - numClasses;
    MNN_ASSERT(maxClassesPerAnchor > 0);
    const int numCategoriesPerAnchor = std::min(maxClassesPerAnchor, numClasses);
    std::vector<float> maxScores;
    maxScores.resize(numBoxes);
    std::vector<int> sortedClassIndices;
    sortedClassIndices.resize(numBoxes * numClasses);
    const auto scoresStartPtr = classPredictions->host<float>();
    // sort scores on every anchor
    for (int idx = 0; idx < numBoxes; ++idx) {
        const auto boxScores = scoresStartPtr + idx * numClassWithBackground + labelOffset;
        auto classIndices    = sortedClassIndices.data() + idx * numClasses;

        std::iota(classIndices, classIndices + numClasses, 0);
        std::partial_sort(classIndices, classIndices + numCategoriesPerAnchor, classIndices + numClasses,
                          [&boxScores](const int i, const int j) { return boxScores[i] > boxScores[j]; });
        maxScores[idx] = boxScores[classIndices[0]];
    }

    std::vector<int> seleted;
    NonMaxSuppressionSingleClasssImpl(decodedBoxes, maxScores.data(), postProcessParam.maxDetections,
                                      postProcessParam.iouThreshold, postProcessParam.nmsScoreThreshold, &seleted);

    const auto decodedBoxesPtr = reinterpret_cast<const BoxCornerEncoding*>(decodedBoxes->host<float>());
    auto detectionBoxesPtr     = reinterpret_cast<BoxCornerEncoding*>(detectionBoxes->host<float>());
    auto detectionClassesPtr   = detectionClass->host<float>();
    auto detectionScoresPtr    = detectionScores->host<float>();
    auto numDetectionsPtr      = numDetections->host<float>();

    int outputBoxIndex = 0;
    for (const auto& selectedIndex : seleted) {
        const float* boxScores  = scoresStartPtr + selectedIndex * numClassWithBackground + labelOffset;
        const int* classIndices = sortedClassIndices.data() + selectedIndex * numClasses;
        for (int col = 0; col < numCategoriesPerAnchor; ++col) {
            int boxOffset                  = numCategoriesPerAnchor * outputBoxIndex + col;
            detectionBoxesPtr[boxOffset]   = decodedBoxesPtr[selectedIndex];
            detectionClassesPtr[boxOffset] = classIndices[col];
            detectionScoresPtr[boxOffset]  = boxScores[classIndices[col]];
            outputBoxIndex++;
        }
    }
    *numDetectionsPtr = outputBoxIndex;
}

CPUDetectionPostProcess::CPUDetectionPostProcess(Backend* bn, const MNN::Op* op) : Execution(bn) {
    auto param = op->main_as_DetectionPostProcessParam();
    param->UnPackTo(&mParam);
    if (mParam.useRegularNMS) {
        MNN_ERROR("TODO, use regular NMS to process decoded boxes!");
        return;
    }
}

ErrorCode CPUDetectionPostProcess::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto boxEncodings    = inputs[0];
    const int numAnchors = boxEncodings->length(1);

    mDecodedBoxes.reset(Tensor::createDevice<float>({numAnchors, 4}));
    auto allocRes = backend()->onAcquireBuffer(mDecodedBoxes.get(), Backend::DYNAMIC);
    if (!allocRes) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mDecodedBoxes.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CPUDetectionPostProcess::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CenterSizeEncoding scaleValues;
    scaleValues.y = mParam.centerSizeEncoding[0];
    scaleValues.x = mParam.centerSizeEncoding[1];
    scaleValues.h = mParam.centerSizeEncoding[2];
    scaleValues.w = mParam.centerSizeEncoding[3];
    _decodeBoxes(inputs[0], inputs[2], scaleValues, mDecodedBoxes.get());

    if (mParam.useRegularNMS) {
        return NOT_SUPPORT;
    } else {
        // perform NMS on max scores
        _NonMaxSuppressionMultiClassFastImpl(mParam, mDecodedBoxes.get(), inputs[1], outputs[0], outputs[1], outputs[2],
                                             outputs[3]);
    }

    return NO_ERROR;
}

class CPUDetectionPostProcessCreator : public CPUBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                        Backend* backend) const override {
        return new CPUDetectionPostProcess(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDetectionPostProcessCreator, OpType_DetectionPostProcess);

} // namespace MNN
