//
//  quanByMSE.cpp
//  MNN
//
//  Created by MNN on 2020/01/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include "DemoUnit.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#include "module/PipelineModule.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <functional>
#include "RandomGenerator.hpp"
#include "ImageNoLabelDataset.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "DataLoader.hpp"
#include "rapidjson/document.h"

#define TRAIN
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;
using namespace MNN::CV;

static ImageDataset::ImageConfig gConfig;
static std::string gImagePath;
static int gChannels;
static int gEpoch;
static std::vector<std::string> gForbid;
static std::vector<int> gInputShape;
static NN::ScaleUpdateMethod gMethod = NN::MovingAverage;
static NN::FeatureScaleStatMethod gFeatureScale = NN::PerChannel;

static bool loadConfig(std::string configPath) {
    std::shared_ptr<ImageDataset::ImageConfig> tempConfig(ImageDataset::ImageConfig::create());
    gConfig = *tempConfig;
    rapidjson::Document document;
    {
        std::ifstream fileNames(configPath.c_str());
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid Config json\n");
            return false;
        }
    }
    auto picObj = document.GetObject();
    if (picObj.HasMember("ScaleUpdateMethod")) {
        std::string type = picObj["ScaleUpdateMethod"].GetString();
        if (type == "Maximum") {
            gMethod = NN::Maximum;
        }
    }
    if (picObj.HasMember("FeatureScaleStatMethod")) {
        std::string type = picObj["FeatureScaleStatMethod"].GetString();
        if (type == "PerTensor") {
            gFeatureScale = NN::PerTensor;
        }
    }
    if (picObj.HasMember("inputShape")) {
        auto shape = picObj["inputShape"].GetArray();
        for (auto iter = shape.begin(); iter != shape.end(); iter++) {
            gInputShape.emplace_back(iter->GetInt());
        }
    }
    auto& config = gConfig;
    config.destFormat = CV::BGR;
    gChannels = 3;
    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}};
            if (formatMap.find(format) != formatMap.end()) {
                config.destFormat = formatMap.find(format)->second;
            }
        }
        if (picObj.HasMember("epoch")) {
            gEpoch = picObj["epoch"].GetInt();
        } else {
            gEpoch = 1;
        }
    }

    if (config.destFormat == GRAY) {
        gChannels = 1;
    }
    std::string imagePath;
    {
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                config.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                config.scale[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("width")) {
            gConfig.resizeWidth = picObj["width"].GetInt();
        }
        if (picObj.HasMember("height")) {
            gConfig.resizeHeight = picObj["height"].GetInt();
        }
        if (picObj.HasMember("path")) {
            gImagePath = picObj["path"].GetString();
        }
    }
    {
        if (picObj.HasMember("skips")) {
            auto array = picObj["skips"].GetArray();
            for (auto iter = array.begin(); iter != array.end(); iter++) {
                gForbid.emplace_back(iter->GetString());
            }
        }
    }
    return true;
}

static VARP _computeLossTrain(VARP target, VARP predict) {
    auto info = target->getInfo();
    if (info->order == NC4HW4) {
        target = _Convert(target, NCHW);
        predict = _Convert(predict, NCHW);
    }
    target = _Reshape(target, {0, -1});
    predict = _Reshape(predict, {0, -1});
    auto loss = _MSE(target, predict);
    return loss;
}
static VARP _computeLoss(VARP target, VARP predict) {
    auto info = target->getInfo();
    if (info->order == NC4HW4) {
        target = _Convert(target, NCHW);
        predict = _Convert(predict, NCHW);
    }
    target = _Reshape(target, {0, -1});
    predict = _Reshape(predict, {0, -1});
    auto loss = _MSE(target, predict);
    return loss;
}
static VARP _computeLossMax(VARP target, VARP predict) {
    auto info = target->getInfo();
    if (info->order == NC4HW4) {
        target = _Convert(target, NCHW);
        predict = _Convert(predict, NCHW);
    }
    target = _Reshape(target, {0, -1});
    predict = _Reshape(predict, {0, -1});
    auto loss = _ReduceMax(_ReduceMax(_Abs(predict - target), {1}));
    return loss;
}
static void dumpVar(VARP var, const char* fileName) {
    std::ofstream output(fileName);
    auto size = var->getInfo()->size;
    auto ptr = var->readMap<float>();
    for (int i=0; i<size; ++i) {
        output << ptr[i] << "\n";
    }
}

static void _test(std::shared_ptr<Module> origin, std::shared_ptr<Module> optmized) {
    auto dataset = ImageNoLabelDataset::create(gImagePath, &gConfig);
    const size_t batchSize  = 1;
    const size_t numWorkers = 0;
    bool shuffle            = false;
    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));
    size_t iterations = (dataset.get<ImageNoLabelDataset>()->size() + batchSize - 1) / batchSize;
    {
        AUTOTIME;
        dataLoader->reset();
        optmized->setIsTraining(false);
        float totalLoss =  0.0f;
        float totalMaxLoss = 0.0f;
        int moveBatchSize = 0;
        int maxBatchIndex = 0;
        std::vector<std::string> errorFileNames;
        auto originFileName = dataset.get<ImageNoLabelDataset>()->files();

        for (int i = 0; i < iterations; i++) {
            // AUTOTIME;
            auto trainData  = dataLoader->next();
            auto example    = trainData[0].first[0];
            moveBatchSize += example->getInfo()->dim[0];
            auto nc4hw4example = _Convert(example, NC4HW4);
            auto target = origin->forward(nc4hw4example);
            auto predict = optmized->forward(nc4hw4example);
            auto loss = _computeLoss(target, predict);
            auto maxLoss = _computeLossMax(target, predict);
            Variable::prepareCompute({loss, maxLoss});
            auto lossValue = loss->readMap<float>()[0];
            auto maxLossValue = maxLoss->readMap<float>()[0];
            if (maxLossValue > totalMaxLoss) {
                maxBatchIndex = i;
                dumpVar(predict, ".predict");
                dumpVar(target, ".target");
            }
            if (maxLossValue > 0.01) {
                errorFileNames.emplace_back(originFileName[i]);
            }
            totalMaxLoss = totalMaxLoss > maxLossValue ? totalMaxLoss : maxLossValue;
            if (i % 10 == 9) {
                std::cout <<"Test " << moveBatchSize << " MSE: " <<lossValue << ", max loss = " << totalMaxLoss << ", Index = " << maxBatchIndex << " \n";
            }
            totalLoss += lossValue * (float)example->getInfo()->dim[0];
        }
        MNN_PRINT("Total Loss MSE: %f\n", totalLoss / moveBatchSize);
        MNN_PRINT("Total Loss %d MAX: %f, Error Number: %d / %d, error index in .temp.error.files\n", maxBatchIndex, totalMaxLoss, (int)errorFileNames.size(), (int)iterations);
        std::ofstream errorIndexesOs(".temp.error.files");
        for (auto& s : errorFileNames) {
            errorIndexesOs << s << "\n";
        }
    }
}

static void _train(std::shared_ptr<Module> origin, std::shared_ptr<Module> optmized, float basicRate, std::string inputName, std::vector<std::string> outputnames, const std::vector<std::string> blockName) {
    auto dataset = ImageNoLabelDataset::create(gImagePath, &gConfig);
    std::shared_ptr<SGD> sgd(new SGD(optmized));
    sgd->setGradBlockName(blockName);
    sgd->setMomentum(1.0f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    const size_t batchSize  = 10;
    const size_t numWorkers = 0;
    bool useTrain = basicRate > 0.0f;
    bool shuffle            = useTrain;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));
    size_t iterations = (dataset.get<ImageNoLabelDataset>()->size() + batchSize - 1) / batchSize;

    for (int epoch = 0; epoch < gEpoch; ++epoch) {
        {
            AUTOTIME;
            dataLoader->reset();
            optmized->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0].first[0];
                moveBatchSize += example->getInfo()->dim[0];
                auto nc4hw4example = _Convert(example, NC4HW4);
                auto predicts = optmized->onForward({nc4hw4example});
                auto targets = origin->onForward({nc4hw4example});
                MNN_ASSERT(targets.size() == predicts.size());
                VARP loss;
                {
                    loss = _computeLossTrain(targets[0], predicts[0]);;
                }
                for (int v=1; v<targets.size(); ++v) {
                    loss = _Maximum(_computeLossTrain(targets[v], predicts[v]), loss);
                }
                float rate   = LrScheduler::inv(basicRate, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                //std::cout << " loss: " << loss->readMap<float>()[0] << "\n";
                //std::cout.flush();
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                }
                if (useTrain) {
                    sgd->step(loss);
                }
            }
        }
        {
            AUTOTIME;
            dataLoader->reset();
            optmized->setIsTraining(false);
            {
                auto forwardInput = _Input({1, gChannels, gConfig.resizeHeight, gConfig.resizeWidth}, NC4HW4);
                forwardInput->setName(inputName);
                auto predict = optmized->onForward({forwardInput});
                MNN_ASSERT(predict.size() == outputnames.size());
                for (int v=0; v<predict.size(); ++v) {
                    predict[v]->setName(outputnames[v]);
                }
                Transformer::turnModelToInfer()->onExecute(predict);
                Variable::save(predict, "temp.quan.mnn");
            }
        }
    }
    _test(origin, optmized);
}
class QuanByMSE : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            MNN_PRINT("usage: ./runTrainDemo.out QuanByMSE /path/to/model quanConfig.json [bits]\n");
            return 0;
        }
        std::string root = argv[2];
        FUNC_PRINT_ALL(root.c_str(), s);
        auto configResult = loadConfig(root);
        if (!configResult) {
            return 0;
        }
        auto varMap      = Variable::loadMap(argv[1]);
        if (varMap.empty()) {
            MNN_ERROR("Can not load model %s\n", argv[1]);
            return 0;
        }
        int bits = 8;
        if (argc > 3) {
            std::istringstream is(argv[3]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        float basicRate = 0.01f;
        if (argc > 4) {
            std::istringstream is(argv[4]);
            is >> basicRate;
        }
        FUNC_PRINT(bits);
        std::vector<std::string> blockName;
        if (argc > 5) {
            std::istringstream is(argv[5]);
            std::string s;
            is >> s;
            blockName.push_back(s);
        }
        FUNC_PRINT_ALL(blockName[0].c_str(), s);
        auto inputOutputs = Variable::getInputAndOutput(varMap);
        auto inputs       = Variable::mapToSequence(inputOutputs.first);
        MNN_ASSERT(inputs.size() == 1);
        auto input = inputs[0];
        std::string inputName = input->name();
        if (gInputShape.size() > 0) {
            input->resize(gInputShape);
        }
        auto inputInfo = input->getInfo();
        MNN_ASSERT(nullptr != inputInfo && inputInfo->order == NC4HW4);
        auto outputs      = Variable::mapToSequence(inputOutputs.second);
        std::vector<std::string> outputNames;
        std::vector<VARP> newOutputs;
        for (int i=0; i<outputs.size(); ++i) {
            auto info = outputs[i]->getInfo();
            if (nullptr == info) {
                MNN_ERROR("Can't compute shape for %s\n", outputs[i]->name().c_str());
                continue;
            }
            if (info->type.code != halide_type_float) {
                continue;
            }
            newOutputs.emplace_back(outputs[i]);
            outputNames.emplace_back(outputs[i]->name());
        }
        if (newOutputs.empty()) {
            MNN_ERROR("No output valid\n");
            return 0;
        }
        {
            auto exe = Executor::getGlobalExecutor();
            BackendConfig config;
            exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 2);
        }
        std::shared_ptr<Module> model(NN::extract(inputs, newOutputs, true));
        NN::turnQuantize(model.get(), bits, gFeatureScale, gMethod);
        std::shared_ptr<Module> originModel(NN::extract(inputs, newOutputs, false));

        _train(originModel, model, basicRate, inputName, outputNames, blockName);
        return 0;
    }
};

class TestMSE : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 3) {
            MNN_PRINT("usage: ./runTrainDemo.out TestMSE /path/to/origin /path/to/quan quanConfig.json \n");
            return 0;
        }
        std::string root = argv[3];
        FUNC_PRINT_ALL(root.c_str(), s);
        auto configResult = loadConfig(root);
        if (!configResult) {
            return 0;
        }
        std::shared_ptr<Module> model0, model1;
        {
            auto varMap      = Variable::loadMap(argv[1]);
            if (varMap.empty()) {
                MNN_ERROR("Can not load model %s\n", argv[1]);
                return 0;
            }
            auto inputOutputs = Variable::getInputAndOutput(varMap);
            auto inputs       = Variable::mapToSequence(inputOutputs.first);
            MNN_ASSERT(inputs.size() == 1);
            auto input = inputs[0];
            std::string inputName = input->name();
            auto inputInfo = input->getInfo();
            MNN_ASSERT(nullptr != inputInfo && inputInfo->order == NC4HW4);
            auto outputs      = Variable::mapToSequence(inputOutputs.second);
            std::vector<std::string> outputNames;
            std::vector<VARP> newOutputs;
            for (int i=0; i<outputs.size(); ++i) {
                auto info = outputs[i]->getInfo();
                if (nullptr == info) {
                    continue;
                }
                if (info->type.code != halide_type_float) {
                    continue;
                }
                newOutputs.emplace_back(outputs[i]);
                outputNames.emplace_back(outputs[i]->name());
            }
            model0.reset(NN::extract(inputs, newOutputs, false));
        }
        {
            auto varMap      = Variable::loadMap(argv[2]);
            if (varMap.empty()) {
                MNN_ERROR("Can not load model %s\n", argv[2]);
                return 0;
            }
            auto inputOutputs = Variable::getInputAndOutput(varMap);
            auto inputs       = Variable::mapToSequence(inputOutputs.first);
            MNN_ASSERT(inputs.size() == 1);
            auto input = inputs[0];
            std::string inputName = input->name();
            auto inputInfo = input->getInfo();
            MNN_ASSERT(nullptr != inputInfo && inputInfo->order == NC4HW4);
            auto outputs      = Variable::mapToSequence(inputOutputs.second);
            std::vector<std::string> outputNames;
            std::vector<VARP> newOutputs;
            for (int i=0; i<outputs.size(); ++i) {
                auto info = outputs[i]->getInfo();
                if (nullptr == info) {
                    continue;
                }
                if (info->type.code != halide_type_float) {
                    continue;
                }
                newOutputs.emplace_back(outputs[i]);
                outputNames.emplace_back(outputs[i]->name());
            }
            model1.reset(NN::extract(inputs, newOutputs, false));
        }
        _test(model0, model1);
        return 0;
    }
};
DemoUnitSetRegister(QuanByMSE, "QuanByMSE");
DemoUnitSetRegister(TestMSE, "TestMSE");
