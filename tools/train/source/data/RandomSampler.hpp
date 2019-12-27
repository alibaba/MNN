//
//  RandomSampler.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RandomSampler_hpp
#define RandomSampler_hpp

#include <vector>
#include "Sampler.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC RandomSampler : public Sampler {
public:
    explicit RandomSampler(size_t size, bool shuffle = true);

    void reset(size_t size) override;

    size_t size() override;

    const std::vector<size_t> indices();

    size_t index();

    std::vector<size_t> next(size_t batchSize) override;

private:
    std::vector<size_t> mIndices;
    size_t mIndex = 0;
    bool mShuffle;
};

} // namespace Train
} // namespace MNN

#endif // RandomSampler
