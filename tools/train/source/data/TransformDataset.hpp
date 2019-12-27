//
//  TransformDataset.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TransformDataset_hpp
#define TransformDataset_hpp

#include <vector>
#include "Dataset.hpp"
#include "Example.hpp"
#include "Transform.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC BatchTransformDataset : public BatchDataset {
public:
    BatchTransformDataset(std::shared_ptr<BatchDataset> dataset, std::shared_ptr<BatchTransform> transform) {
        MNN_ASSERT(dataset != nullptr);
        mDataset   = dataset;
        mTransform = transform;
    }

    std::vector<Example> getBatch(std::vector<size_t> indices) override {
        auto batch = mDataset->getBatch(indices);
        if (mTransform != nullptr) {
            batch = mTransform->transformBatch(std::move(batch));
        }

        return batch;
    }

    size_t size() override {
        return mDataset->size();
    }

private:
    std::shared_ptr<BatchDataset> mDataset;
    std::shared_ptr<BatchTransform> mTransform;
};

class MNN_PUBLIC TransformDataset : public Dataset {
public:
    TransformDataset(std::shared_ptr<Dataset> dataset, std::shared_ptr<Transform> transform) {
        MNN_ASSERT(dataset != nullptr);
        mDataset   = dataset;
        mTransform = transform;
    }

    Example get(size_t index) override {
        auto example = mDataset->get(index);
        if (mTransform != nullptr) {
            example = mTransform->transformExample(std::move(example));
        }

        return example;
    }

    size_t size() override {
        return mDataset->size();
    }

private:
    std::shared_ptr<Dataset> mDataset;
    std::shared_ptr<Transform> mTransform;
};

} // namespace Train
} // namespace MNN

#endif
