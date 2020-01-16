//
//  Sampler.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Sampler_hpp
#define Sampler_hpp

#include <MNN/MNNDefine.h>
#include <vector>

namespace MNN {
namespace Train {

class MNN_PUBLIC Sampler {
public:
    virtual ~Sampler() = default;

    virtual void reset(size_t size) = 0;

    virtual size_t size() = 0;

    virtual std::vector<size_t> next(size_t batchSize) = 0;
};

} // namespace Train
} // namespace MNN

#endif // Sampler
