//
//  testQuanModel.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "rapidjson/document.h"
#define STB_IMAGE_IMPLEMENTATION
#include <dirent.h>
#include <sys/stat.h>
#include "stb_image.h"

using namespace MNN;
using namespace MNN::CV;
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_ERROR("Usage: ./testQuanModel.out float.mnn quan.mnn preTreatConfig.json\n");
        return 0;
    }
    const char* originFile     = argv[1];
    const char* preTreatConfig = argv[3];
    const char* quanFile       = argv[2];
    FUNC_PRINT_ALL(originFile, s);
    FUNC_PRINT_ALL(preTreatConfig, s);
    FUNC_PRINT_ALL(quanFile, s);
    int width  = 0;
    int height = 0;
    std::string imagePath;
    std::shared_ptr<ImageProcess> process;
    {
        rapidjson::Document document;
        std::ifstream fileNames(preTreatConfig);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
        auto picObj = document.GetObject();
        ImageProcess::Config config;
        config.destFormat = BGR;
        {
            if (picObj.HasMember("format")) {
                auto format = picObj["format"].GetString();
                static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}};
                if (formatMap.find(format) != formatMap.end()) {
                    config.destFormat = formatMap.find(format)->second;
                }
            }
        }
        config.sourceFormat = RGBA;
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
                    config.normal[cur++] = iter->GetFloat();
                }
            }
            if (picObj.HasMember("width")) {
                width = picObj["width"].GetInt();
            }
            if (picObj.HasMember("height")) {
                height = picObj["height"].GetInt();
            }
            if (picObj.HasMember("path")) {
                imagePath = picObj["path"].GetString();
            }
        }
        process.reset(ImageProcess::create(config));
    }

    std::vector<std::string> images;
    DIR* root = opendir(imagePath.c_str());
    if (root == NULL) {
        MNN_ERROR("Open %s Failed\n", imagePath.c_str());
        return 0;
    }
    struct dirent* ent = readdir(root);
    while (ent != NULL) {
        if (ent->d_name[0] != '.') {
            const std::string fileName = imagePath + "/" + ent->d_name;
            images.push_back(fileName);
        }
        ent = readdir(root);
    }

    ScheduleConfig scheConfig;
    scheConfig.type = MNN_FORWARD_CPU;
    scheConfig.numThread = 4;
    std::unique_ptr<Interpreter> originModel(Interpreter::createFromFile(originFile));
    std::unique_ptr<Interpreter> quanModel(Interpreter::createFromFile(quanFile));
    auto originSession = originModel->createSession(scheConfig);
    auto quanSession   = quanModel->createSession(scheConfig);
    auto originInput   = originModel->getSessionInput(originSession, nullptr);
    auto originOutput  = originModel->getSessionOutput(originSession, nullptr);
    std::shared_ptr<Tensor> originOutputHostTensor(new Tensor(originOutput, originOutput->getDimensionType()));
    {
        originModel->resizeTensor(originInput, 1, originInput->channel(), height, width);
        originModel->resizeSession(originSession);
        originModel->releaseModel();
    }
    auto quanInput  = quanModel->getSessionInput(quanSession, nullptr);
    auto quanOutput = quanModel->getSessionOutput(quanSession, nullptr);
    {
        quanModel->resizeTensor(quanInput, 1, quanInput->channel(), height, width);
        quanModel->resizeSession(quanSession);
        quanModel->releaseModel();
    }
    std::shared_ptr<Tensor> quanOutputHostTensor(new Tensor(quanOutput, quanOutput->getDimensionType()));

    int top1Result = 0;
    int top5Result = 0;
    int number     = 0;

    const int step = 500;
    for (auto& image : images) {
        int originalWidth, originalHeight, comp;
        auto bitmap32bits = stbi_load(image.c_str(), &originalWidth, &originalHeight, &comp, 4);
        if (nullptr == bitmap32bits) {
            MNN_ERROR("Open %s error\n", image.c_str());
            continue;
        }
        number++;

        MNN::CV::Matrix trans;
        trans.setScale((float)(originalWidth - 1) / (float)(width - 1),
                       (float)(originalHeight - 1) / (float)(height - 1));
        process->setMatrix(trans);
        process->convert(bitmap32bits, originalWidth, originalHeight, 0, quanInput);
        process->convert(bitmap32bits, originalWidth, originalHeight, 0, originInput);

        stbi_image_free(bitmap32bits);

        originModel->runSession(originSession);
        quanModel->runSession(quanSession);

        originOutput->copyToHostTensor(originOutputHostTensor.get());
        quanOutput->copyToHostTensor(quanOutputHostTensor.get());

        auto size = originOutputHostTensor->elementSize();
        std::vector<std::pair<float, int>> originOutput(size);
        std::vector<std::pair<float, int>> quanOutput(size);
        for (int i = 0; i < size; ++i) {
            originOutput[i] = std::make_pair(originOutputHostTensor->host<float>()[i], i);
            quanOutput[i]   = std::make_pair(quanOutputHostTensor->host<float>()[i], i);
        }
        std::sort(originOutput.rbegin(), originOutput.rend());
        std::sort(quanOutput.rbegin(), quanOutput.rend());
//        MNN_PRINT("Top1, Quan: %d , %f | Origin: %d, %f\n", quanOutput[0].second, quanOutput[0].first,
//                  originOutput[0].second, originOutput[0].first);
        if (quanOutput[0].second == originOutput[0].second) {
            top1Result++;
        }
        bool matchTop5 = false;
        for (int i = 0; i < 5; ++i) {
            if (quanOutput[i].second == originOutput[0].second) {
                matchTop5 = true;
                break;
            }
        }
        if (matchTop5) {
            top5Result++;
        }
        if (number % step == 0) {
            MNN_PRINT("Currnet Number: %d, Top1: %f, Top5: %f\n", number, 100.0f*(float)top1Result / (float)number, 100.0f*(float)top5Result / (float)number);
        }
    }

    MNN_PRINT("Top1: %f\n", (float)top1Result / (float)number);
    MNN_PRINT("Top5: %f\n", (float)top5Result / (float)number);
}
