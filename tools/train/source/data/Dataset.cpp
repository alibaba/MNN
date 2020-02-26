//
//  Dataset.cpp
//  MNN
//
//  Created by MNN on 2020/02/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Dataset.hpp"
namespace MNN {
namespace Train {

DataLoader* DatasetPtr::createLoader(const int batchSize, const bool stack, const bool shuffle, const int numWorkers) {
    return DataLoader::makeDataLoader(mDataset, batchSize, stack, shuffle, numWorkers);
}
} // namespace Train
} // namespace MNN
