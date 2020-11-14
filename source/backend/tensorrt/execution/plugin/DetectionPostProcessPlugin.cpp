//
//  DetectionPostProcessPlugin.cpp
//  MNN
//
//  Created by MNN on b'2020/08/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DetectionPostProcessPlugin.hpp"
namespace MNN {

DetectionPostProcessPlugin::DetectionPostProcessPlugin(const Op *op, const MNNTRTPlugin::Plugin *plugin) {
    auto detection = plugin->main_as_DetectionPostProcessInfo();
    const int numAnchors0 = detection->numAnchors0();
    
    mDecodedBoxes = std::make_shared<CudaBind<float>>(numAnchors0*4);

    mScaleValues = std::make_shared<CudaBind<float>>(4);
    auto status = cudaMemcpy(static_cast<void*>(mScaleValues->mPtr), static_cast<const void*>(detection->scaleValues()->data()), sizeof(float) * 4, cudaMemcpyHostToDevice);
    CUASSERT(status);
    mNumBoxes = detection->numBoxes();
    mBoxCoordNum = detection->boxCoordNum();
    mAnchorsCoordNum = detection->anchorsCoordNum();
    mNumAnchors1 = detection->numAnchors1();

    mNumClassWithBackground = detection->numClassWithBackground();

    mNumClasses = detection->numClasses();
    mMaxClassesPerAnchor = detection->maxClassesPerAnchor();

    mMaxScores = std::make_shared<CudaBind<float>>(mNumBoxes);
    mSortedClassIndices = std::make_shared<CudaBind<int>>(mNumBoxes*mNumClasses);

    mMaxDetections = detection->maxDetections();
    mIouThreshold = detection->iouThreshold();
    mNmsScoreThreshold = detection->nmsScoreThreshold();

    mOutputNum = std::min(mMaxDetections, mNumBoxes);
    mSelected = std::make_shared<CudaBind<int>>(mOutputNum);

    mSelectedSize = std::make_shared<CudaBind<int>>(1);
    cudaMemset(mSelectedSize->mPtr, 0, sizeof(int));

    mCandidate = std::make_shared<CudaBind<Candidate>>(mNumBoxes);
    mCandidatePriorityQueue = std::make_shared<CudaBind<Candidate>>(mNumBoxes);

    mOutputBoxIndex = std::make_shared<CudaBind<float>>(1);
    cudaMemset(mOutputBoxIndex->mPtr, 0, sizeof(float));

}

DetectionPostProcessPlugin::~DetectionPostProcessPlugin() {
    // Do nothgin
}

int DetectionPostProcessPlugin::onEnqueue(int batchSize, const void *const *inputs, void **outputs, void* workspace, nvinfer1::DataType dataType, cudaStream_t stream) {

    decodeBoxes(dataType, mNumBoxes, inputs, outputs, mScaleValues->mPtr, mDecodedBoxes->mPtr, mNumBoxes, mBoxCoordNum, mAnchorsCoordNum, mNumAnchors1);

    maxScores(dataType, mNumBoxes, inputs, outputs, mNumClassWithBackground, (int*)mSortedClassIndices->mPtr, mNumClasses, (float*)mMaxScores->mPtr, mMaxClassesPerAnchor);

    NMSSingleClasss((float*)mDecodedBoxes->mPtr, (const float*)mMaxScores->mPtr, mMaxDetections, mIouThreshold, mNmsScoreThreshold, (int*)mSelected->mPtr, (int*)mSelectedSize->mPtr, mNumBoxes, mOutputNum,(Candidate*)mCandidate->mPtr, (Candidate*)mCandidatePriorityQueue->mPtr);
    
    int selectSize;
    auto status = cudaMemcpy(static_cast<void*>(&selectSize), static_cast<const void*>(mSelectedSize->mPtr), sizeof(int), cudaMemcpyDeviceToHost);
    CUASSERT(status);

    const int labelOffset = mNumClassWithBackground - mNumClasses;
    const int numCategoriesPerAnchor = std::min(mMaxClassesPerAnchor, mNumClasses);
    cudaMemset(outputs[3], 0, sizeof(float));
    setOutput(selectSize, (BoxCornerEncoding*)(mDecodedBoxes->mPtr), (BoxCornerEncoding*)outputs[0], (float*)outputs[1], (float*)outputs[2], (float*)outputs[3], (float*)inputs[1], mNumClassWithBackground, labelOffset, (int*)mSortedClassIndices->mPtr, mNumClasses, numCategoriesPerAnchor, (int*)mSelected->mPtr);
    return cudaPeekAtLastError();
}

}; // namespace MNN
