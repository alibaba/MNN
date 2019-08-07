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
#include "logkit.h"

// Given distribution P and Q, KL-Divergence is
// Sum(P[i] * log(P[i] / Q[i]))
static float _klDivergence(const std::vector<float>& candidateDis, const std::vector<float>& expandedDis) {
    float result   = 0.0f;
    const int size = candidateDis.size();

    for (int i = 0; i < size; ++i) {
        if (candidateDis[i] != 0) {
            if(expandedDis[i] == 0){
                result += 1.0f;
            }else{
                result += (candidateDis[i] * std::log(candidateDis[i] / expandedDis[i]));
            }
        }
    }

    return result;
}

TensorStatistic::TensorStatistic(const MNN::Tensor* tensor, std::string method, const std::string& name, int binNumber, GET_THRESHOLD_METHOD thresholdMethod)
    : mOriginTensor(tensor), mName(name), mBinNumber(binNumber), mThresholdMethod(thresholdMethod) {
    MNN_ASSERT(tensor->dimensions() == 4);
    if (method == "KL") {
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
        std::fill(c.begin(), c.end(), 1.0e-07);
    }
    // MNN_PRINT("==> %s max: %f\n", mName.c_str(),std::max(fabsf(mRangePerChannel[0].second), fabsf(mRangePerChannel[0].first)));
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

void TensorStatistic::setChannelWise(bool mergeChannel){
    mMergeChannel = mergeChannel;
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
        auto scale     = ((float)threshold + 0.5) / mIntervals[0] / 127.0f;
        // MNN_PRINT("==> %s == %d, %f, %f\n", mName.c_str(),threshold, 1.0f / mIntervals[0], scale * 127.0f);
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

std::vector<float> TensorStatistic::computeScaleADMM() {
    std::vector<float> scaleValue(mOriginTensor->channel(), 0.0f);
    
    const int count = mOriginTensor->elementSize();
    float max = 0;
    const float bound = 127;
    const float* originData = mOriginTensor->host<float>();
    
    for (int i = 0; i < count; i++) {
        float absData = std::fabs(originData[i]);
        if (absData > max) {
            max = absData;
        }
    }
    float alpha = max / (bound * 2.5);

    DLOG(INFO) << "alpha init: " << alpha;

    const int maxStep = 300;
    float sum1 = 0;
    float sum2 = 0;
    float invAlpha;

    for (int i = 0; i < maxStep; i++) {
        sum1 = 0;
        sum2 = 0;
        invAlpha = 1 / alpha;

        for (int i = 0; i < count; i++){
            auto origin = originData[i];
            auto dataQuant = std::roundf(origin * invAlpha);
            dataQuant = std::fmin(bound, std::fmax(-bound, dataQuant));
            sum1 += (dataQuant * origin);
            sum2 += (dataQuant * dataQuant);
        }

        alpha = sum1 / sum2;
    }
    DLOG(INFO) << "alpha final: " << alpha;

    std::fill(scaleValue.begin(), scaleValue.end(), alpha);
    return scaleValue;
}