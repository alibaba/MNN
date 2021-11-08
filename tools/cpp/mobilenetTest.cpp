//
//  mobilenetV1Test.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {

    if (argc < 3) {
        MNN_PRINT("Usage: ./mobilenetTest.out model.mnn input.jpg [forwardType] [precision] [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = 4;
    if (argc > 3) {
        config.type = (MNNForwardType)::atoi(argv[3]);
    }

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    if (argc > 4) {
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)::atoi(argv[4]);
    }
    config.backendConfig = &backendConfig;
    MNN_PRINT("model:%s, input image:%s, forwardType:%d, precision:%d\n", argv[1], argv[2], config.type, (int)backendConfig.precision);
    Session* session = net->createSession(config);

    Tensor* inputTensor  = net->getSessionInput(session, NULL);
    Tensor* outputTensor = net->getSessionOutput(session, NULL);

    Tensor inputTensorUser(inputTensor, Tensor::DimensionType::TENSORFLOW);
    Tensor outputTensorUser(outputTensor, outputTensor->getDimensionType());

    //image preproccess
    {
        int netInputHeight = inputTensorUser.height();
        int netInputWidth  = inputTensorUser.width();

        int imageChannel, imageWidth, imageHeight;
        unsigned char* inputImage = stbi_load(argv[2], &imageWidth,
                                              &imageHeight, &imageChannel, 4);

        Matrix trans;
        trans.setScale(1.0 / imageWidth, 1.0 / imageHeight);
        trans.postRotate(0, 0.5f, 0.5f);
        trans.postScale(netInputWidth, netInputHeight);
        trans.invert(&trans);

        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3]     = {103.94f, 116.78f, 123.68f};
        float normals[3]  = {0.017f, 0.017f, 0.017f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat = RGB;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert(inputImage, imageWidth, imageHeight, 0, &inputTensorUser);

        stbi_image_free(inputImage);
    }

    //run
    {
        AUTOTIME;
        inputTensor->copyFromHostTensor(&inputTensorUser);
        net->runSession(session);
        outputTensor->copyToHostTensor(&outputTensorUser);
    }


    //get predict labels
    {

        std::vector<std::string> words;
        if (argc > 5) {
            std::ifstream inputOs(argv[5]);
            std::string line;
            while (std::getline(inputOs, line)) {
                words.emplace_back(line);
            }
        }

        MNN_PRINT("output size:%d\n", outputTensorUser.elementSize());
        auto type = outputTensorUser.getType();

        auto size = outputTensorUser.elementSize();
        std::vector<std::pair<int, float>> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputTensorUser.host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1) {
            auto values = outputTensorUser.host<uint8_t>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        // Find Max
        std::sort(tempValues.begin(), tempValues.end(),
                  [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });

        int length = size > 10 ? 10 : size;
        if (words.empty()) {
            for (int i = 0; i < length; ++i) {
                MNN_PRINT("%d, %f\n", tempValues[i].first, tempValues[i].second);
            }
        } else {
            for (int i = 0; i < length; ++i) {
                MNN_PRINT("%s: %f\n", words[tempValues[i].first].c_str(), tempValues[i].second);
            }
        }
    }

    return 0;
}
