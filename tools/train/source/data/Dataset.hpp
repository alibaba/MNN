//
//  Dataset.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Dataset_hpp
#define Dataset_hpp

#include <MNN/MNNDefine.h>
#include <vector>
#include "Example.hpp"
#include "DataLoader.hpp"

namespace MNN {
namespace Train {
struct MNN_PUBLIC DatasetPtr {
public:
    std::shared_ptr<BatchDataset> mDataset;

    DataLoader* createLoader(
                              const int batchSize,
                              const bool stack = true,
                              const bool shuffle = true,
                              const int numWorkers = 0);
    ~ DatasetPtr() = default;
    template<typename T>
    T* get() const {
        return (T*)mDataset.get();
    }
};

class MNN_PUBLIC BatchDataset {
public:
    virtual ~BatchDataset() = default;

    // get batch using given indices
    virtual std::vector<Example> getBatch(std::vector<size_t> indices) = 0;

    // size of the dataset
    virtual size_t size() = 0;
};

class MNN_PUBLIC Dataset : public BatchDataset {
public:
    // return a specific example with given index
    virtual Example get(size_t index) = 0;

    std::vector<Example> getBatch(std::vector<size_t> indices) {
        std::vector<Example> batch;
        batch.reserve(indices.size());
        for (const auto i : indices) {
            batch.emplace_back(get(i));
        }
        MNN_ASSERT(batch.size() != 0);
        return batch;
    }
};

} // namespace Train
} // namespace MNN

#endif /* Dataset_hpp */
