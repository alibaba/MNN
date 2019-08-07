//
//  TensorStatistic.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <memory>
#include <vector>
#include "Tensor.hpp"
#include <string>

enum GET_THRESHOLD_METHOD {
    THRESHOLD_MAX = 0,
    THRESHOLD_KL  = 1,
};

class TensorStatistic {
public:
    TensorStatistic(const MNN::Tensor* tensor, std::string method, const std::string& name, int binNumber = 2048, GET_THRESHOLD_METHOD thresholdMethod = THRESHOLD_KL);
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
};
