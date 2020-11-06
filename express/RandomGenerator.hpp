//
//  RandomGenerator.hpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RandomGenerator_hpp
#define RandomGenerator_hpp

#include <MNN/MNNDefine.h>
#include <random>

namespace MNN {
namespace Express {

class MNN_PUBLIC RandomGenerator {
private:
    RandomGenerator(int seed = std::random_device()()) {
        mSeed = seed;
        mGenerator.seed(mSeed);
    }

    ~RandomGenerator() = default;

    RandomGenerator(RandomGenerator &);

    RandomGenerator &operator=(const RandomGenerator &);

private:
    int mSeed;
    std::mt19937 mGenerator;

public:
    static std::mt19937 &generator(int seed = std::random_device()()) {
        static RandomGenerator rng(seed);
        return rng.mGenerator;
    }
};

} // namespace Express
} // namespace MNN

#endif // RandomGenerator_hpp