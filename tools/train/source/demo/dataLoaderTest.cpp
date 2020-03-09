//
//  dataLoaderTest.cpp
//  MNN
//
//  Created by MNN on 2019/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "LambdaTransform.hpp"
#include "RandomSampler.hpp"
#include "Sampler.hpp"
#include "StackTransform.hpp"
#include "Transform.hpp"
#include "TransformDataset.hpp"

using namespace std;
using namespace MNN::Train;
using namespace MNN;

class DataLoaderTest : public DemoUnit {
public:
    // this function is an example to use the lambda transform
    // here we use lambda transform to normalize data from 0~255 to 0~1
    static Example func(Example example) {
        // an easier way to do this
        auto cast = _Cast(example.first[0], halide_type_of<float>());
        return {{_Multiply(cast, _Const(1.0f / 255.0f)), example.first[1]}, {example.second}};
    }

    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out DataLoaderTest /path/to/unzipped/mnist/data/" << endl;
            return 0;
        }

        const int testCount = 6;
        int passedTestCount = 0;

        std::string root = argv[1];

        // train data loader
        const size_t trainDatasetSize = 60000;
        auto trainDataset             = MnistDataset::create(root, MnistDataset::Mode::TRAIN);

        auto trainSampler = std::make_shared<RandomSampler>(trainDataset.get<MnistDataset>()->size());

        const size_t trainBatchSize  = 7;
        const size_t trainNumWorkers = 4;
        auto trainConfig             = std::make_shared<DataLoaderConfig>(trainBatchSize, trainNumWorkers);

        DataLoader trainDataLoader(trainDataset.mDataset, trainSampler, trainConfig);

        auto images                 = trainDataset.get<MnistDataset>()->images();
        auto labels                 = trainDataset.get<MnistDataset>()->labels();
        const int32_t kImageRows    = 28;
        const int32_t kImageColumns = 28;

        const size_t iterations = trainDatasetSize / trainBatchSize;

        auto samplerIndices = trainSampler->indices();
        sort(samplerIndices.begin(), samplerIndices.end());
        for (int i = 0; i < samplerIndices.size(); i++) {
            MNN_ASSERT(samplerIndices[i] == i);
        }

        for (int i = 0; i < iterations; i++) {
            auto trainData = trainDataLoader.next();

            for (int j = 0; j < trainData.size(); j++) {
                auto index = int(trainData[j].first[1]->readMap<float>()[0]);

                auto data  = trainData[j].first[0]->readMap<uint8_t>();
                auto label = trainData[j].second[0]->readMap<uint8_t>();

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    MNN_ASSERT(data[k] == trueData[k]);
                }
                MNN_ASSERT(label[0] == trueLabel[0]);
            }
        }
        trainDataLoader.clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        // the lambda transform for one example, we also can do it in batch
        auto trainLambdaTransform    = std::make_shared<LambdaTransform>(func);
        auto trainLambdaTransDataset = std::make_shared<BatchTransformDataset>(trainDataset.mDataset, trainLambdaTransform);

        DataLoader trainLambdaDataLoader(trainLambdaTransDataset, trainSampler, trainConfig);

        samplerIndices = trainSampler->indices();
        sort(samplerIndices.begin(), samplerIndices.end());
        for (int i = 0; i < samplerIndices.size(); i++) {
            MNN_ASSERT(samplerIndices[i] == i);
        }

        for (int i = 0; i < iterations; i++) {
            auto trainData = trainLambdaDataLoader.next();

            for (int j = 0; j < trainData.size(); j++) {
                auto index = int(trainData[j].first[1]->readMap<float>()[0]);

                auto data  = trainData[j].first[0]->readMap<float>();
                auto label = trainData[j].second[0]->readMap<uint8_t>();

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    MNN_ASSERT(fabs(data[k] - (trueData[k] / 255.0f)) < 1e-6);
                }
                MNN_ASSERT(label[0] == trueLabel[0]);
            }
        }
        trainLambdaDataLoader.clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
        auto trainStackTransform    = std::make_shared<StackTransform>();
        auto trainStackTransDataset = std::make_shared<BatchTransformDataset>(trainDataset.mDataset, trainStackTransform);

        DataLoader trainStackDataLoader(trainStackTransDataset, trainSampler, trainConfig);

        samplerIndices = trainSampler->indices();
        sort(samplerIndices.begin(), samplerIndices.end());
        for (int i = 0; i < samplerIndices.size(); i++) {
            MNN_ASSERT(samplerIndices[i] == i);
        }

        for (int i = 0; i < iterations; i++) {
            auto trainData = trainStackDataLoader.next();

            auto data  = trainData[0].first[0]->readMap<uint8_t>();
            auto label = trainData[0].second[0]->readMap<uint8_t>();

            for (int j = 0; j < trainBatchSize; j++) {
                auto index = int(trainData[0].first[1]->readMap<float>()[j]);

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    int dataIndex = j * (kImageRows * kImageColumns) + k;
                    MNN_ASSERT(data[dataIndex] == trueData[k]);
                }
                MNN_ASSERT(label[j] == trueLabel[0]);
            }
        }
        trainStackDataLoader.clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        // here we test Lambda + Stack
        auto trainLambdaStackTransDataset =
            std::make_shared<BatchTransformDataset>(trainLambdaTransDataset, trainStackTransform);

        DataLoader trainLambdaStackDataLoader(trainLambdaStackTransDataset, trainSampler, trainConfig);

        samplerIndices = trainSampler->indices();
        sort(samplerIndices.begin(), samplerIndices.end());
        for (int i = 0; i < samplerIndices.size(); i++) {
            MNN_ASSERT(samplerIndices[i] == i);
        }

        for (int i = 0; i < iterations; i++) {
            auto trainData = trainLambdaStackDataLoader.next();

            auto data  = trainData[0].first[0]->readMap<float>();
            auto label = trainData[0].second[0]->readMap<uint8_t>();

            for (int j = 0; j < trainBatchSize; j++) {
                auto index = int(trainData[0].first[1]->readMap<float>()[j]);

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    int dataIndex = j * (kImageRows * kImageColumns) + k;
                    MNN_ASSERT(fabs(data[dataIndex] - (trueData[k] / 255.0f)) < 1e-6);
                }
                MNN_ASSERT(label[j] == trueLabel[0]);
            }
        }
        trainLambdaStackDataLoader.clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        // here we test Stack + Lambda
        auto trainStackLambdaTransDataset =
            std::make_shared<BatchTransformDataset>(trainStackTransDataset, trainLambdaTransform);

        DataLoader trainStackLamdaDataLoader(trainStackLambdaTransDataset, trainSampler, trainConfig);

        samplerIndices = trainSampler->indices();
        sort(samplerIndices.begin(), samplerIndices.end());
        for (int i = 0; i < samplerIndices.size(); i++) {
            MNN_ASSERT(samplerIndices[i] == i);
        }

        for (int i = 0; i < iterations; i++) {
            auto trainData = trainStackLamdaDataLoader.next();

            auto data  = trainData[0].first[0]->readMap<float>();
            auto label = trainData[0].second[0]->readMap<uint8_t>();

            for (int j = 0; j < trainBatchSize; j++) {
                auto index = int(trainData[0].first[1]->readMap<float>()[j]);

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    int dataIndex = j * (kImageRows * kImageColumns) + k;
                    MNN_ASSERT(fabs(data[dataIndex] - (trueData[k] / 255.0f)) < 1e-6);
                }
                MNN_ASSERT(label[j] == trueLabel[0]);
            }
        }
        trainStackLamdaDataLoader.clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        // test makeDataLoader
        auto madeDataLoader = std::shared_ptr<DataLoader>(DataLoader::makeDataLoader(
            trainDataset.mDataset, {nullptr, trainStackTransform, nullptr, trainLambdaTransform, nullptr}, 7));

        for (int i = 0; i < iterations; i++) {
            auto trainData = madeDataLoader->next();

            auto data  = trainData[0].first[0]->readMap<float>();
            auto label = trainData[0].second[0]->readMap<uint8_t>();

            for (int j = 0; j < trainBatchSize; j++) {
                auto index = int(trainData[0].first[1]->readMap<float>()[j]);

                auto trueData  = images->readMap<uint8_t>() + kImageRows * kImageColumns * index;
                auto trueLabel = labels->readMap<uint8_t>() + index;

                for (int k = 0; k < kImageRows * kImageColumns; k++) {
                    int dataIndex = j * (kImageRows * kImageColumns) + k;
                    MNN_ASSERT(fabs(data[dataIndex] - (trueData[k] / 255.0f)) < 1e-6);
                }
                MNN_ASSERT(label[j] == trueLabel[0]);
            }
        }
        madeDataLoader->clean();

        passedTestCount++;
        cout << "[" << passedTestCount << " / " << testCount << "] passed." << endl;

        return 0;
    }
};

DemoUnitSetRegister(DataLoaderTest, "DataLoaderTest");
