//
//  classficationTopkEval.cpp
//  MNN
//
//  Created by MNN on 2019/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "rapidjson/document.h"
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

#define TOPK 5
#define TOTAL_CLASS_NUM 1001

void computeTopkAcc(const std::vector<int>& groundTruthId, const std::vector<std::pair<int, float>>& sortedResult,
                    int index, int* top1, int* topk) {
    const int label = groundTruthId[index];
    if (sortedResult[0].first == label) {
        (*top1)++;
    }
    for (int i = 0; i < TOPK; ++i) {
        if (label == sortedResult[i].first) {
            (*topk)++;
            break;
        }
    }
}

int runEvaluation(const char* modelPath, const char* preTreatConfig) {
    int height, width;
    std::string imagePath;
    std::string groundTruthIdFile;
    rapidjson::Document document;
    {
        std::ifstream fileNames(preTreatConfig);
        std::ostringstream output;
        output << fileNames.rdbuf();
        auto outputStr = output.str();
        document.Parse(outputStr.c_str());
        if (document.HasParseError()) {
            MNN_ERROR("Invalid json\n");
            return 0;
        }
    }
    auto picObj = document.GetObject();
    ImageProcess::Config config;
    config.filterType = BILINEAR;
    // defalut input image format
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
        if (picObj.HasMember("imagePath")) {
            imagePath = picObj["imagePath"].GetString();
        }
        if (picObj.HasMember("groundTruthId")) {
            groundTruthIdFile = picObj["groundTruthId"].GetString();
        }
    }

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));

    std::shared_ptr<Interpreter> classficationInterpreter(Interpreter::createFromFile(modelPath));
    ScheduleConfig classficationEvalConfig;
    classficationEvalConfig.type      = MNN_FORWARD_CPU;
    classficationEvalConfig.numThread = 4;
    auto classficationSession         = classficationInterpreter->createSession(classficationEvalConfig);
    auto inputTensor                  = classficationInterpreter->getSessionInput(classficationSession, nullptr);
    auto shape                        = inputTensor->shape();
    // the model has not input dimension
    if(shape.size() == 0){
        shape.resize(4);
        shape[0] = 1;
        shape[1] = 3;
        shape[2] = height;
        shape[3] = width;
    }
    // set batch to be 1
    shape[0] = 1;
    classficationInterpreter->resizeTensor(inputTensor, shape);
    classficationInterpreter->resizeSession(classficationSession);

    auto outputTensor = classficationInterpreter->getSessionOutput(classficationSession, nullptr);

    // read ground truth label id
    std::vector<int> groundTruthId;
    {
        std::ifstream inputOs(groundTruthIdFile);
        std::string line;
        while (std::getline(inputOs, line)) {
            groundTruthId.emplace_back(std::atoi(line.c_str()));
        }
    }

    // read images file path
    int count = 0;
    std::vector<std::string> files;
    {
#if defined(_MSC_VER)
        WIN32_FIND_DATA ffd;
        HANDLE hFind = INVALID_HANDLE_VALUE;
        hFind = FindFirstFile(imagePath.c_str(), &ffd);
        if (INVALID_HANDLE_VALUE == hFind) {
            printf("Error to open %s\n", imagePath.c_str());
            return 0;
        }
        do {
            if(INVALID_FILE_ATTRIBUTES != GetFileAttributes(ffd.cFileName) && GetLastError() != ERROR_FILE_NOT_FOUND) {
                files.push_back(ffd.cFileName);
            }
        } while (FindNextFile(hFind, &ffd) != 0);
        FindClose(hFind);
#else
        struct stat s;
        lstat(imagePath.c_str(), &s);
        struct dirent* filename;
        DIR* dir;
        dir = opendir(imagePath.c_str());
        while ((filename = readdir(dir)) != nullptr) {
            if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
                continue;
            }
            files.push_back(filename->d_name);
            count++;
        }
#endif
        std::cout << "total: " << count << std::endl;
        std::sort(files.begin(), files.end());
    }

    if (count != groundTruthId.size()) {
        MNN_ERROR("The number of input images is not same with ground truth id\n");
        return 0;
    }

    int test = 0;
    int top1 = 0;
    int topk = 0;

    const int outputTensorSize = outputTensor->elementSize();
    if (outputTensorSize != TOTAL_CLASS_NUM) {
        MNN_ERROR("Change the total class number, such as the result number of tensorflow mobilenetv1/v2 is 1001\n");
        return 0;
    }

    std::vector<std::pair<int, float>> sortedResult(outputTensorSize);
    for (const auto& file : files) {
        const auto img = imagePath + file;
        int h, w, channel;
        auto inputImage = stbi_load(img.c_str(), &w, &h, &channel, 4);
        if (!inputImage) {
            MNN_ERROR("Can't open %s\n", img.c_str());
            return 0;
        }

        // input image transform
        Matrix trans;
        // choose resize or crop
        // resize method
        // trans.setScale((float)(w-1) / (width-1), (float)(h-1) / (height-1));
        // crop method
        trans.setTranslate(16.0f, 16.0f);
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)inputImage, h, w, 0, inputTensor);
        stbi_image_free(inputImage);
        classficationInterpreter->runSession(classficationSession);

        {
            // default float value
            auto outputDataPtr = outputTensor->host<float>();
            for (int i = 0; i < outputTensorSize; ++i) {
                sortedResult[i] = std::make_pair(i, outputDataPtr[i]);
            }
            std::sort(sortedResult.begin(), sortedResult.end(),
                      [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });
        }
        computeTopkAcc(groundTruthId, sortedResult, test, &top1, &topk);
        test++;
        MNN_PRINT("==> tested: %f, Top1: %f, Topk: %f\n", (float)test / (float)count * 100.0,
                  (float)top1 / (float)test * 100.0, (float)topk / (float)test * 100.0);
    }

    return 0;
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./classficationTopkEval.out model.mnn preTreatConfig.json\n");
    }

    const auto modelPath          = argv[1];
    const auto preTreatConfigFile = argv[2];

    runEvaluation(modelPath, preTreatConfigFile);

    return 0;
}
