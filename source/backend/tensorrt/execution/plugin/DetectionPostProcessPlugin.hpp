//
//  DetectionPostProcessPlugin.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DetectionPostProcessPlugin_hpp
#define DetectionPostProcessPlugin_hpp
#include "CommonPlugin.hpp"
#include <MNN/MNNDefine.h>

using namespace std;
namespace MNN {

struct CenterSizeEncoding {
    float y;
    float x;
    float h;
    float w;
};

struct BoxCornerEncoding {
    float ymin;
    float xmin;
    float ymax;
    float xmax;
};

struct Candidate {
    int index;
    int boxIndex;
    float score;
};

class DetectionPostProcessPlugin : public CommonPlugin::Enqueue {
public:
    DetectionPostProcessPlugin(const Op* op, const MNNTRTPlugin::Plugin* plugin);
    ~DetectionPostProcessPlugin();
    
    virtual int onEnqueue(int batchSize, const void* const* inputs, void** outputs, void*, nvinfer1::DataType dataType,
                          cudaStream_t stream) override;

    void decodeBoxes(nvinfer1::DataType dataType, const int count, const void *const * inputs, const void *const * outputs, const void * scaleValues, void * decodeBoxes, int numBoxes, int boxCoordNum, int anchorsCoordNum, int numAnchors1);

    void maxScores(nvinfer1::DataType dataType, const int count, const void *const * inputs, const void *const * outputs, int numClassWithBackground, int* sortedClassIndicesPtr, int numClasses, float* maxScores, int maxClassesPerAnchor);
    
    void NMSSingleClasss(float* decodedBoxesPtr, const float* scoresPtr, int maxDetections, float iouThreshold, float scoreThreshold, int* selectedPtr, int* selectedSize, int numBoxes, int outputNum, Candidate* candidate, Candidate* mCandidatePriorityQueue);

    void setOutput(const int selectSize, const BoxCornerEncoding* decodedBoxesPtr, BoxCornerEncoding* detectionBoxesPtr, float* detectionClassesPtr, float* detectionScoresPtr, float* numDetectionsPtr, const float* scoresStartPtr, int numClassWithBackground, int labelOffset, int* sortedClassIndicesPtr, int numClasses, int numCategoriesPerAnchor, int* selectedPtr);

private:

    int mAnchorsCnt;
    int mMaxBatchSize;
    std::shared_ptr<CudaBind<float>> mDecodedBoxes;
    std::shared_ptr<CudaBind<float>> mScaleValues;
    std::shared_ptr<CudaBind<int>> mSortedClassIndices;
    std::shared_ptr<CudaBind<float>> mMaxScores;

    std::shared_ptr<CudaBind<int>> mSelected;
    std::shared_ptr<CudaBind<int>> mSelectedSize;

    std::shared_ptr<CudaBind<Candidate>> mCandidate;
    std::shared_ptr<CudaBind<Candidate>> mCandidatePriorityQueue;
    std::shared_ptr<CpuBind<float>> mDebug;

    std::shared_ptr<CudaBind<float>> mOutputBoxIndex;

    int mNumBoxes;
    int mBoxCoordNum;
    int mAnchorsCoordNum;
    int mNumAnchors1;
    int mNumClassWithBackground;
    int mNumClasses;
    int mMaxClassesPerAnchor;
    int mMaxDetections;
    float mIouThreshold;
    float mNmsScoreThreshold;
    int mOutputNum;
};

} // namespace MNN
#endif /* DetectionPostProcessPlugin_hpp */
