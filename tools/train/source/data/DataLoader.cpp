//
//  DataLoader.cpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DataLoader.hpp"

namespace MNN {
namespace Train {

std::vector<Example> DataLoader::next() {
    if (mConfig->numWorkers == 0) {
        auto batchIndices = mSampler->next(mConfig->batchSize);
        MNN_ASSERT(batchIndices.size() != 0); // the sampler is exhausted, should reset the data loader
        if (mConfig->dropLast && batchIndices.size() < mConfig->batchSize) {
            MNN_ASSERT(false); // the sampler is exhausted
        }
        auto batch = mDataset->getBatch(batchIndices);
        return batch;
    } else {
        auto batch = mDataQueue->pop();
        prefetch(1);
        return batch;
    }
}

void DataLoader::prefetch(size_t nJobs) {
    MNN_ASSERT(mJobs != nullptr);
    for (int i = 0; i < nJobs; i++) {
        auto batchIndices = mSampler->next(mConfig->batchSize);
        Job j;
        j.job = batchIndices;
        if (batchIndices.size() != 0) {
            if (mConfig->dropLast && batchIndices.size() < mConfig->batchSize) {
                // drop the job
            } else {
                mJobs->push(std::move(j)); // the job may be empty when sampler is exhausted
            }
        }
    }
}

void DataLoader::workerThread() {
    while (true) {
        auto currentJob = mJobs->pop();
        if (currentJob.quit) {
            break;
        }
        // make sure there are no empty jobs, so that there are no empty batch
        MNN_ASSERT(currentJob.job.size() != 0);
        auto batch = mDataset->getBatch(currentJob.job);
        mDataQueue->push(std::move(batch));
    }
}

void DataLoader::join() {
    for (int i = 0; i < mConfig->numWorkers; i++) {
        Job j;
        j.quit = true;
        mJobs->push(std::move(j));
    }
    for (auto& worker : mWorkers) {
        worker.join();
    }
}

void DataLoader::reset() {
    clean();

    if (mConfig->numWorkers > 0) {
        prefetch(mConfig->numJobs);
        for (int i = 0; i < mConfig->numWorkers; i++) {
            mWorkers.emplace_back([&] { workerThread(); });
        }
    }
}

void DataLoader::clean() {
    if (mJobs != nullptr) {
        join();
        mWorkers.clear();
        mJobs->clear();
        mDataQueue->clear();
    }
    // should reset sampler before prefetch
    mSampler->reset(mSampler->size());
}

std::shared_ptr<DataLoader> DataLoader::makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                                       std::vector<std::shared_ptr<BatchTransform>> transforms,
                                                       const int batchSize, const bool shuffle, const int numWorkers) {
    if (transforms.size() > 0) {
        std::shared_ptr<BatchTransformDataset> transDataset = nullptr;
        bool flag                                           = true;
        for (int i = 0; i < transforms.size(); i++) {
            if (transforms[i] != nullptr) {
                if (flag) {
                    transDataset = std::make_shared<BatchTransformDataset>(dataset, transforms[i]);
                    flag         = false;
                } else {
                    transDataset = std::make_shared<BatchTransformDataset>(transDataset, transforms[i]);
                }
            }
        }

        if (transDataset != nullptr) {
            auto sampler = std::make_shared<RandomSampler>(transDataset->size(), shuffle);
            auto config  = std::make_shared<DataLoaderConfig>(batchSize, numWorkers);

            return std::make_shared<DataLoader>(transDataset, sampler, config);
        } else {
            auto sampler = std::make_shared<RandomSampler>(dataset->size(), shuffle);
            auto config  = std::make_shared<DataLoaderConfig>(batchSize, numWorkers);

            return std::make_shared<DataLoader>(dataset, sampler, config);
        }
    } else {
        auto sampler = std::make_shared<RandomSampler>(dataset->size(), shuffle);
        auto config  = std::make_shared<DataLoaderConfig>(batchSize, numWorkers);

        return std::make_shared<DataLoader>(dataset, sampler, config);
    }
}

} // namespace Train
} // namespace MNN
