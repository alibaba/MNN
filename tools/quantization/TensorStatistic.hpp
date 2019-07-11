//
//  TensorStatistic.hpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <vector>
#include "Tensor.hpp"
#include <memory>

class TensorStatistic {
public:
    TensorStatistic(const MNN::Tensor* tensor, int binNumber);
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

    bool mMergeChannel = false;
};
