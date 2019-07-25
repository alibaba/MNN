//
//  TensorStatistic.cpp
//  MNN
//
//  Created by MNN on 2019/06/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TensorStatistic.hpp"
#include <math.h>
#include <algorithm>
#include <cmath>
#include "MNNDefine.h"

// Given distribution P and Q, KL-Divergence is
// Sum(P[i] * log(P[i] / Q[i]))
static float _klDivergence(const std::vector<float>& candidateDis, const std::vector<float>& expandedDis) {
    float result   = 0.0f;
    const int size = candidateDis.size();

    for (int i = 0; i < size; ++i) {
        result += (candidateDis[i] * std::log(candidateDis[i] / expandedDis[i]));
    }

    return result;
}

static void _smoothDistribution(std::vector<float>& distribution) {
    const float eps = 1e-3;
    const int size  = distribution.size();
    int zeroNum     = 0;
    std::for_each(distribution.begin(), distribution.end(), [&](float n) {
        if (n == 0) {
            zeroNum++;
        }
    });
    const int nonZeroNum = size - zeroNum;
    const float eps1     = (float)zeroNum / (float)nonZeroNum * eps;

    std::for_each(distribution.begin(), distribution.end(), [=](float& n) {
        if (n == 0) {
            n = eps;
        } else {
            n -= eps1;
        }
        MNN_ASSERT(n > 0);
    });
}

TensorStatistic::TensorStatistic(const MNN::Tensor* tensor, int binNumber, GET_THRESHOLD_METHOD thresholdMethod)
    : mOriginTensor(tensor), mBinNumber(binNumber), mThresholdMethod(thresholdMethod) {
    MNN_ASSERT(tensor->dimensions() == 4);
    auto channel = tensor->channel();
    mRangePerChannel.resize(channel);
    for (auto& iter : mRangePerChannel) {
        iter.first  = 100000.0f;  // Min Init
        iter.second = -100000.0f; // Max Init
    }
    mIntervals.resize(channel);
    mValidChannel.resize(channel);
    mHostTensor.reset(new MNN::Tensor(tensor, MNN::Tensor::CAFFE));
    mDistribution.resize(channel);
    for (int c = 0; c < mDistribution.size(); ++c) {
        mDistribution[c].resize(mBinNumber);
    }
    bool isLittleAmountData = tensor->width() * tensor->height() < 100;
    if (isLittleAmountData) {
        mThresholdMethod = THRESHOLD_MAX;
    }
}
void TensorStatistic::updateRange() {
    if (mUpdatedRangeFlags) {
        return;
    }
    mUpdatedRangeFlags = true;
    mOriginTensor->copyToHostTensor(mHostTensor.get());
    int batch   = mHostTensor->batch();
    int channel = mHostTensor->channel();
    int width   = mHostTensor->width();
    int height  = mHostTensor->height();
    auto area   = width * height;

    for (int n = 0; n < batch; ++n) {
        auto dataBatch = mHostTensor->host<float>() + n * mHostTensor->stride(0);
        for (int c = 0; c < channel; ++c) {
            int cIndex = c;
            if (mMergeChannel) {
                cIndex = 0;
            }
            auto minValue    = mRangePerChannel[cIndex].first;
            auto maxValue    = mRangePerChannel[cIndex].second;
            auto dataChannel = dataBatch + c * mHostTensor->stride(1);
            for (int v = 0; v < area; ++v) {
                minValue = std::min(minValue, dataChannel[v]);
                maxValue = std::max(maxValue, dataChannel[v]);
            }
            mRangePerChannel[cIndex].first  = minValue;
            mRangePerChannel[cIndex].second = maxValue;
        }
    }
}

void TensorStatistic::resetDistribution() {
    for (int i = 0; i < mIntervals.size(); ++i) {
        int cIndex = i;
        if (mMergeChannel) {
            cIndex = 0;
        }
        auto maxValue         = std::max(fabsf(mRangePerChannel[cIndex].second), fabsf(mRangePerChannel[cIndex].first));
        mValidChannel[cIndex] = maxValue > 0.00001f;
        mIntervals[cIndex]    = 0.0f;
        if (mValidChannel[cIndex]) {
            mIntervals[cIndex] = (float)mBinNumber / maxValue;
        }
    }
    for (auto& c : mDistribution) {
        std::fill(c.begin(), c.end(), 0.0f);
    }
}
void TensorStatistic::updateDistribution() {
    if (mUpdatedDistributionFlag) {
        return;
    }
    mUpdatedDistributionFlag = true;
    mOriginTensor->copyToHostTensor(mHostTensor.get());
    int batch   = mHostTensor->batch();
    int channel = mHostTensor->channel();
    int width   = mHostTensor->width();
    int height  = mHostTensor->height();
    auto area   = width * height;

    for (int n = 0; n < batch; ++n) {
        auto dataBatch = mHostTensor->host<float>() + n * mHostTensor->stride(0);
        for (int c = 0; c < channel; ++c) {
            int cIndex = c;
            if (mMergeChannel) {
                cIndex = 0;
            }
            if (!mValidChannel[cIndex]) {
                continue;
            }
            auto multi       = mIntervals[cIndex];
            auto target      = mDistribution[cIndex].data();
            auto dataChannel = dataBatch + c * mHostTensor->stride(1);
            for (int v = 0; v < area; ++v) {
                auto data = dataChannel[v];
                if (data == 0) {
                    continue;
                }
                int index = static_cast<int>(fabs(data) * multi);
                index     = std::min(index, mBinNumber - 1);
                target[index] += 1.0f;
            }
        }
    }
}

void TensorStatistic::setThresholdMethod(GET_THRESHOLD_METHOD thresholdMethod) {
    mThresholdMethod = thresholdMethod;
}

int TensorStatistic::_computeThreshold(const std::vector<float>& distribution) {
    const int targetBinNums = 128;
    int threshold           = targetBinNums;

    if (mThresholdMethod == THRESHOLD_KL) {
        float minKLDivergence   = 10000.0f;
        float afterThresholdSum = 0.0f;
        std::for_each(distribution.begin() + targetBinNums, distribution.end(),
                      [&](float n) { afterThresholdSum += n; });
        for (int i = targetBinNums; i < mBinNumber; ++i) {
            std::vector<float> quantizedDistribution(targetBinNums);
            std::vector<float> candidateDistribution(i);
            std::vector<float> expandedDistribution(i);
            std::copy(distribution.begin(), distribution.begin() + i, candidateDistribution.begin());
            candidateDistribution[i - 1] += afterThresholdSum;
            afterThresholdSum -= distribution[i];

            const float binInterval = (float)i / (float)targetBinNums;

            // merge i bins to target bins
            for (int j = 0; j < targetBinNums; ++j) {
                const float start = j * binInterval;
                const float end   = start + binInterval;

                const int leftUpper = static_cast<int>(std::ceil(start));
                if (leftUpper > start) {
                    const float leftScale = leftUpper - start;
                    quantizedDistribution[j] += leftScale * distribution[leftUpper - 1];
                }
                const int rightLower = static_cast<int>(std::floor(end));
                if (rightLower < end) {
                    const float rightScale = end - rightLower;
                    quantizedDistribution[j] += rightScale * distribution[rightLower];
                }
                std::for_each(distribution.begin() + leftUpper, distribution.begin() + rightLower,
                              [&](float n) { quantizedDistribution[j] += n; });
            }
            // expand target bins to i bins
            for (int j = 0; j < targetBinNums; ++j) {
                const float start   = j * binInterval;
                const float end     = start + binInterval;
                float count         = 0;
                const int leftUpper = static_cast<int>(std::ceil(start));
                float leftScale     = 0.0f;
                if (leftUpper > start) {
                    leftScale = leftUpper - start;
                    if (distribution[leftUpper - 1] != 0) {
                        count += leftScale;
                    }
                }
                const int rightLower = static_cast<int>(std::floor(end));
                float rightScale     = 0.0f;
                if (rightLower < end) {
                    rightScale = end - rightLower;
                    if (distribution[rightLower] != 0) {
                        count += rightScale;
                    }
                }

                std::for_each(distribution.begin() + leftUpper, distribution.begin() + rightLower, [&](float n) {
                    if (n != 0) {
                        count += 1;
                    }
                });

                if (count == 0) {
                    continue;
                }
                const float toExpandValue = quantizedDistribution[j] / count;
                if (leftUpper > start && distribution[leftUpper - 1] != 0) {
                    expandedDistribution[leftUpper - 1] += toExpandValue * leftScale;
                }
                if (rightLower < end && distribution[rightLower] != 0) {
                    expandedDistribution[rightLower] += toExpandValue * rightScale;
                }

                for (int k = leftUpper; k < rightLower; ++k) {
                    if (distribution[k] != 0) {
                        expandedDistribution[k] += toExpandValue;
                    }
                }
            }

            _smoothDistribution(candidateDistribution);
            _smoothDistribution(expandedDistribution);
            const float curKL = _klDivergence(candidateDistribution, expandedDistribution);
            // std::cout << "=====> KL: " << i << " ==> " << curKL << std::endl;
            if (curKL < minKLDivergence) {
                minKLDivergence = curKL;
                threshold       = i;
            }
        }
    } else if (mThresholdMethod == THRESHOLD_MAX) {
        threshold = mBinNumber - 1;
    } else {
        // TODO, support other method
        MNN_ASSERT(false);
    }
    return threshold;
}

std::vector<float> TensorStatistic::finishAndCompute() {
    std::vector<float> scaleValue(mDistribution.size(), 0.0f);
    if (mMergeChannel) {
        if (!mValidChannel[0]) {
            return scaleValue;
        }
        float sum          = 0.0f;
        auto& distribution = mDistribution[0];
        std::for_each(distribution.begin(), distribution.end(), [&](float n) { sum += n; });
        std::for_each(distribution.begin(), distribution.end(), [sum](float& n) { n /= sum; });

        auto threshold = _computeThreshold(distribution);
        auto scale     = ((float)threshold) / mIntervals[0] / 127.0f;
        std::fill(scaleValue.begin(), scaleValue.end(), scale);
        return scaleValue;
    }
    for (int c = 0; c < mDistribution.size(); ++c) {
        if (!mValidChannel[c]) {
            continue;
        }
        float sum          = 0.0f;
        auto& distribution = mDistribution[c];
        std::for_each(distribution.begin(), distribution.end(), [&](float n) { sum += n; });
        std::for_each(distribution.begin(), distribution.end(), [sum](float& n) { n /= sum; });

        auto threshold = _computeThreshold(distribution);
        scaleValue[c]  = ((float)threshold + 0.5) / mIntervals[c] / 127.0;
    }
    return scaleValue;
}
