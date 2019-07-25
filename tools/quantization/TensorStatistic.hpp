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

enum GET_THRESHOLD_METHOD {
    THRESHOLD_MAX = 0,
    THRESHOLD_KL  = 1,
};

class TensorStatistic {
public:
    TensorStatistic(const MNN::Tensor* tensor, int binNumber, GET_THRESHOLD_METHOD thresholdMethod = THRESHOLD_KL);
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

    std::vector<float> finishAndCompute();

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
    GET_THRESHOLD_METHOD mThresholdMethod = THRESHOLD_KL;
};
