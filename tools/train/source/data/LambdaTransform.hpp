//
//  LambdaTransform.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LambdaTransform_hpp
#define LambdaTransform_hpp

#include <functional>
#include "Example.hpp"
#include "Transform.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC BatchLambdaTransform : public BatchTransform {
public:
    explicit BatchLambdaTransform(std::function<std::vector<Example>(std::vector<Example>)> f) {
        func_ = f;
    }

    std::vector<Example> transformBatch(std::vector<Example> batch) override {
        return func_(std::move(batch));
    }

private:
    std::function<std::vector<Example>(std::vector<Example>)> func_;
};

class MNN_PUBLIC LambdaTransform : public Transform {
public:
    explicit LambdaTransform(std::function<Example(Example)> f) {
        mFunc = f;
    }

    Example transformExample(Example example) override {
        return mFunc(std::move(example));
    }

private:
    std::function<Example(Example)> mFunc;
};

} // namespace Train
} // namespace MNN

#endif // LambdaTransform_hpp