//
//  DataLoader.hpp
//  MNN
//
//  Created by MNN on 2019/11/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DataLoader_hpp
#define DataLoader_hpp

#include <string>
#include <thread>
#include <vector>
#include "BlockingQueue.hpp"
#include "DataLoaderConfig.hpp"
#include "Dataset.hpp"
#include "Example.hpp"
#include "LambdaTransform.hpp"
#include "RandomSampler.hpp"
#include "Sampler.hpp"
#include "StackTransform.hpp"
#include "Transform.hpp"
#include "TransformDataset.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC DataLoader {
public:
    DataLoader(std::shared_ptr<BatchDataset> dataset, std::shared_ptr<Sampler> sampler,
               std::shared_ptr<DataLoaderConfig> config) {
        mDataset = dataset;
        mSampler = sampler;
        mConfig  = config;
        if (mConfig->numJobs > 0) {
            mJobs      = std::make_shared<BlockingQueue<Job>>(mConfig->numJobs);
            mDataQueue = std::make_shared<BlockingQueue<std::vector<Example>>>(mConfig->numJobs);
            prefetch(mConfig->numJobs);
            for (int i = 0; i < mConfig->numWorkers; i++) {
                mWorkers.emplace_back([&] { workerThread(); });
            }
        }
    }

    /*
     When use Windows v141 toolset to compile class having vector of non-copyable element (std::thread, for example),
     copy constructor (or assignment operator) must be deleted explicity, otherwise compile will failed.
     */
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator = (const DataLoader&) = delete;

    virtual ~DataLoader() {
        join();
    };

    void prefetch(size_t nJobs);

    void workerThread();

    void join();

    std::vector<Example> next();

    void reset();

    void clean();

    static std::shared_ptr<DataLoader> makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                                      std::vector<std::shared_ptr<BatchTransform>> transforms,
                                                      const int batchSize, const bool shuffle = true,
                                                      const int numWorkers = 0);

private:
    struct Job {
        std::vector<size_t> job;
        bool quit = false;
    };
    std::shared_ptr<BatchDataset> mDataset;
    std::shared_ptr<Sampler> mSampler;
    std::shared_ptr<DataLoaderConfig> mConfig;
    std::shared_ptr<BlockingQueue<Job>> mJobs;
    std::shared_ptr<BlockingQueue<std::vector<Example>>> mDataQueue;
    std::vector<std::thread> mWorkers;
};

} // namespace Train
} // namespace MNN

#endif // DataLoader_hpp
