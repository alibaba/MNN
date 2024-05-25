//
//  TensorStatistic.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <vector>
#include <MNN/Tensor.hpp>
#include <string>

enum GET_THRESHOLD_METHOD {
    THRESHOLD_MAX = 0,
    THRESHOLD_KL  = 1,
};

class TensorStatistic {
public:
    TensorStatistic(const MNN::Tensor* tensor, std::string method, const std::string& name, float featureClampValue, int binNumber = 2048, GET_THRESHOLD_METHOD thresholdMethod = THRESHOLD_KL);
    ~TensorStatistic() {
        // Do nothing
    }

    void resetUpdatedDistributionFlag() {
        mUpdatedDistributionFlag = false;
    }
    void resetUpdatedRangeFlags() {
        mUpdatedRangeFlags = false;
    }
    void updateRange();
    void resetDistribution();
    void updateDistribution();

    void setThresholdMethod(GET_THRESHOLD_METHOD thresholdMethod);

    std::pair<float, int8_t> finishAndCompute();

    // only this one for ADMM
    std::pair<float, int8_t> computeScaleADMM();

    std::string name() {
        return mName;
    }

    bool visited() {
        return mVisited;
    }

    void setVisited(bool visited) {
        mVisited = visited;
    }

    std::pair<std::vector<float>, float> fakeQuantFeature();
    float computeDistance(std::vector<float> fakeQuantedFeature);

private:
    int _computeThreshold(const std::vector<float>& distribution);
    // <minVal, maxVal> for every channel for the Tensor
    std::pair<float, float> mRange;
    // mBinNumber / maxValue: the number of bin for range 1
    float mInterval;
    // if the i-th channel's maxValue > 0.00001f, mValidChannel[i] is true
    bool mValid;
    // [c * mBinNumber]: store every channel's distribution using bin
    std::vector<float> mDistribution;

    std::shared_ptr<MNN::Tensor> mHostTensor;
    // the Tensor
    const MNN::Tensor* mOriginTensor;
    // bin number for distribution
    int mBinNumber;
    // has update or not, assert update once
    bool mUpdatedDistributionFlag = false;
    bool mUpdatedRangeFlags       = false;

    std::string mName;
    GET_THRESHOLD_METHOD mThresholdMethod = THRESHOLD_KL;
    bool mVisited = false;
    float mScale;
    int8_t mZeroPoint = 0;
    float mFeatureClampValue = 127.0f;
};
