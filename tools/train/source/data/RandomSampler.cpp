//
//  RandomSampler.cpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "RandomSampler.hpp"
#include <algorithm>
#include <iterator>
#include <random>
#include "Distributions.hpp"
#include "RandomGenerator.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {

RandomSampler::RandomSampler(size_t size, bool shuffle) {
    mIndices.reserve(size);
    for (int i = 0; i < size; i++) {
        mIndices.emplace_back(i);
    }

    mShuffle = shuffle;
    if (mShuffle) {
        std::shuffle(mIndices.begin(), mIndices.end(), RandomGenerator::generator());
    }
}

void RandomSampler::reset(size_t size) {
    mIndices.clear();
    mIndices.reserve(size);
    for (int i = 0; i < size; i++) {
        mIndices.emplace_back(i);
    }

    if (mShuffle) {
        std::shuffle(mIndices.begin(), mIndices.end(), RandomGenerator::generator());
    }

    mIndex = 0;
}

size_t RandomSampler::size() {
    return mIndices.size();
}

const std::vector<size_t> RandomSampler::indices() {
    return mIndices;
}

size_t RandomSampler::index() {
    return mIndex;
}

std::vector<size_t> RandomSampler::next(size_t batchSize) {
    MNN_ASSERT(mIndex <= mIndices.size());

    auto remainIndices = mIndices.size() - mIndex;
    if (remainIndices == 0) {
        return {};
    }

    std::vector<size_t> batchIndex(std::min(batchSize, remainIndices));
    std::copy(mIndices.begin() + mIndex, mIndices.begin() + mIndex + batchIndex.size(), batchIndex.begin());

    mIndex += batchIndex.size();

    return batchIndex;
}

} // namespace Train
} // namespace MNN
