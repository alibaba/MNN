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
    void setChannelWise(bool mergeChannel);

    std::vector<float> finishAndCompute();

    // only this one for ADMM
    std::vector<float> computeScaleADMM();

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
    std::vector<std::pair<float, float>> mRangePerChannel;
    std::vector<float> mIntervals;
    std::vector<bool> mValidChannel;
    std::vector<std::vector<float>> mDistribution;

    std::shared_ptr<MNN::Tensor> mHostTensor;
    const MNN::Tensor* mOriginTensor;
    int mBinNumber;
    bool mUpdatedDistributionFlag = false;
    bool mUpdatedRangeFlags       = false;

    bool mMergeChannel                    = true;
    std::string mName;
    GET_THRESHOLD_METHOD mThresholdMethod = THRESHOLD_KL;
    bool mVisited = false;
    std::vector<float> mScales;
    float mFeatureClampValue = 127.0f;
};
