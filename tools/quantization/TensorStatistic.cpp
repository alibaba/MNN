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
#include <MNN/MNNDefine.h>
#include "logkit.h"
#include "core/TensorUtils.hpp"

// Given distribution P and Q, KL-Divergence is
// Sum(P[i] * log(P[i] / Q[i]))
static float _klDivergence(const std::vector<float>& candidateDis, const std::vector<float>& expandedDis) {
    float result   = 0.0f;
    const int size = candidateDis.size();

    for (int i = 0; i < size; ++i) {
        if (candidateDis[i] != 0) {
            if (expandedDis[i] == 0) {
                result += 1.0f;
            } else {
                result += (candidateDis[i] * std::log(candidateDis[i] / expandedDis[i]));
            }
        }
    }

    return result;
}

TensorStatistic::TensorStatistic(const MNN::Tensor* tensor, std::string method, const std::string& name, float featureClampValue, int binNumber,
                                 GET_THRESHOLD_METHOD thresholdMethod)
    : mOriginTensor(tensor), mName(name), mBinNumber(binNumber), mThresholdMethod(thresholdMethod), mFeatureClampValue(featureClampValue) {
    // MNN_ASSERT(tensor->dimensions() == 4);
    if (method == "KL") {
        auto channel = tensor->channel();
        mRange.first  = 100000.0f;  // Min Init
        mRange.second = -100000.0f; // Max Init
        mHostTensor.reset(new MNN::Tensor(tensor, MNN::Tensor::CAFFE));
        mDistribution.resize(mBinNumber);
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
    auto tmpTensor = mOriginTensor;
    bool res = mOriginTensor->copyToHostTensor(mHostTensor.get());
    if (res) {
        tmpTensor = mHostTensor.get();
    }
    int size = tmpTensor->elementSize();
    float* dataPtr = tmpTensor->host<float>();
    auto minValue = mRange.first;
    auto maxValue = mRange.second;
    for (int i = 0; i < size; ++i) {
        minValue = std::min(minValue, dataPtr[i]);
        maxValue = std::max(maxValue, dataPtr[i]);
    }
    mRange.first = minValue;
    mRange.second = maxValue;
//    if (mRange.first > 0.0f) {
//        mRange.first = 0.0f;
//    }
//    if (mRange.second < 0.0f) {
//        mRange.second = 0.0f;
//    }
    mVisited = true;
}

void TensorStatistic::resetDistribution() {
    auto maxValue         = std::max(fabsf(mRange.second), fabsf(mRange.first));
    mValid = maxValue > 0.00001f;
    mInterval    = 0.0f;
    if (mValid) {
        mInterval = (float)mBinNumber / maxValue;
    }
    std::fill(mDistribution.begin(), mDistribution.end(), 1.0e-07);
    // MNN_PRINT("==> %s max: %f\n", mName.c_str(),std::max(fabsf(mRangePerChannel[0].second),
    // fabsf(mRangePerChannel[0].first)));
}
void TensorStatistic::updateDistribution() {
    if (mUpdatedDistributionFlag) {
        return;
    }
    mUpdatedDistributionFlag = true;
    auto tmpTensor = mOriginTensor;
    bool res = mOriginTensor->copyToHostTensor(mHostTensor.get());
    if (res) {
        tmpTensor = mHostTensor.get();
    }
//    float midValue = (mRange.second + mRange.first) / 2.0f;
    float midValue = 0.f;
    auto ptr = tmpTensor->host<float>();
    for (int i = 0; i < tmpTensor->elementSize(); ++i) {
        auto data = ptr[i] - midValue;
        if (abs(data) <= 1e-6) {
            continue;
        }
        int index = static_cast<int>(fabs(data) * mInterval);
        index = std::min(index, mBinNumber - 1);
        mDistribution[index] += 1.0f;
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

std::pair<float, int8_t> TensorStatistic::finishAndCompute() {
    if (!mValid) {
        return std::make_pair(0.f, 0);
    }
    float sum          = 0.0f;
    std::for_each(mDistribution.begin(), mDistribution.end(), [&](float n) { sum += n; });
    std::for_each(mDistribution.begin(), mDistribution.end(), [sum](float& n) { n /= sum; });

    auto threshold = _computeThreshold(mDistribution);
    mScale     = ((float)threshold + 0.5) / mInterval / mFeatureClampValue;
    // MNN_PRINT("==> %s == %d, %f, %f\n", mName.c_str(),threshold, 1.0f / mIntervals[0], mScale * mFeatureClampValue);
    float midValue = (mRange.second + mRange.first) / 2.0f;
//    mZeroPoint = fmaxf(fminf(-midValue / mScale, 127.0f), -127.0f);
    mZeroPoint = 0;

    return std::make_pair(mScale, mZeroPoint);
}

std::pair<float, int8_t> TensorStatistic::computeScaleADMM() {
    const int count         = mOriginTensor->elementSize();
    float max               = 0;
    const float bound       = mFeatureClampValue;
    const float* originData = mOriginTensor->host<float>();

    for (int i = 0; i < count; i++) {
        float absData = std::fabs(originData[i]);
        if (absData > max) {
            max = absData;
        }
    }
    float alpha = max / (bound * 2.5);

    // DLOG(INFO) << "alpha init: " << alpha;

    const int maxStep = 300;
    float sum1        = 0;
    float sum2        = 0;
    float invAlpha;

    for (int i = 0; i < maxStep; i++) {
        sum1     = 0;
        sum2     = 0;
        invAlpha = 1 / alpha;

        for (int i = 0; i < count; i++) {
            auto origin    = originData[i];
            auto dataQuant = std::roundf(origin * invAlpha);
            dataQuant      = std::fmin(bound, std::fmax(-bound, dataQuant));
            sum1 += (dataQuant * origin);
            sum2 += (dataQuant * dataQuant);
        }

        alpha = sum1 / sum2;
    }
    // DLOG(INFO) << "alpha final: " << alpha;
    mScale = alpha;
    mVisited = true;
    mZeroPoint = 0;
    return std::make_pair(mScale, mZeroPoint);
}

std::pair<std::vector<float>, float> TensorStatistic::fakeQuantFeature() {
    const int count         = mOriginTensor->elementSize();
    const float bound       = mFeatureClampValue;
    float* originData = mOriginTensor->host<float>();
    const float scale = mScale;
    std::vector<float> fakeQuantedFeature;
    int overflowCount = 0;

    for (int i = 0; i < count; i++) {
        float dataQuant = std::roundf(originData[i] / scale) + mZeroPoint;
        dataQuant      = std::fmin(bound, std::fmax(-bound, dataQuant));
        float dataDequant = (dataQuant - mZeroPoint) * scale;

        originData[i] = dataDequant;
        fakeQuantedFeature.emplace_back(dataDequant);

        if (std::fabs(std::fabs(dataQuant) - bound) < 1e-6) {
            overflowCount++;
        }
    }

    float overflowRatio = overflowCount / float(count);
    auto result = std::make_pair(fakeQuantedFeature, overflowRatio);

    mVisited = true;
    return result;
}

float TensorStatistic::computeDistance(std::vector<float> fakeQuantedFeature) {
    const int count         = mOriginTensor->elementSize();
    CHECK_EQ(count, fakeQuantedFeature.size()) << "feature size error";
    const float bound       = mFeatureClampValue;
    float* originData = mOriginTensor->host<float>();
    float axbSum = 0.0f;
    float a2Sum = 0.0f;
    float b2Sum = 0.0f;

    for (int i = 0; i < count; i++) {
        axbSum += (originData[i] * fakeQuantedFeature[i]);
        a2Sum += (originData[i] * originData[i]);
        b2Sum += (fakeQuantedFeature[i] * fakeQuantedFeature[i]);
    }

    float cosDis = axbSum / std::sqrt(a2Sum) / std::sqrt(b2Sum);

    mVisited = true;
    return cosDis;
}
