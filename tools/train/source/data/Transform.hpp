//
//  Transform.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Transform_hpp
#define Transform_hpp

#include <vector>
#include "Example.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC BatchTransform {
public:
    virtual ~BatchTransform() = default;

    virtual std::vector<Example> transformBatch(std::vector<Example> batch) = 0;
};

class MNN_PUBLIC Transform : public BatchTransform {
public:
    virtual Example transformExample(Example example) = 0;

    std::vector<Example> transformBatch(std::vector<Example> batch) {
        std::vector<Example> outputBatch;
        outputBatch.reserve(batch.size());
        for (auto& example : batch) {
            outputBatch.emplace_back(transformExample(std::move(example)));
        }
        return outputBatch;
    }
};

} // namespace Train
} // namespace MNN

#endif // Transform_hpp
