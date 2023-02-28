//
//  pictureRecognition_batch.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
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
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"

using namespace MNN::CV;
using namespace MNN;
using namespace std;

// behave like python split
vector<string> split(const string sourceStr, string splitChar = " ") {
    vector<string> result;
    int pos   = 0;
    int start = 0;

    while ((pos = sourceStr.find(splitChar, start)) != string::npos) {
        result.emplace_back(sourceStr.substr(start, pos - start));
        start = pos + splitChar.size();
    }

    if (start < sourceStr.size()) {
        result.emplace_back(sourceStr.substr(start));
    }

    return result;
};

#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }

static void dumpTensor2File(const Tensor* tensor, const char* file) {
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
#ifdef MNN_USE_SSE
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (static_cast<int>(data[x + z * width]) - 128) << "\t";
            }
            outputOs << "\n";
        }
#else
        DUMP_CHAR_DATA(int8_t);
#endif
    }
}
static void _initDebug() {
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const OperatorInfo*) {
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const OperatorInfo* info) {
        auto opName = info->name();
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            auto expectTensor = new MNN::Tensor(ntensor, outDimType);
            ntensor->copyToHostTensor(expectTensor);

            auto tensor = expectTensor;

            std::ostringstream outputFileName;
            auto opCopyName = opName;
            for (int j = 0; j < opCopyName.size(); ++j) {
                if (opCopyName[j] == '/') {
                    opCopyName[j] = '_';
                }
            }
            if (tensor->dimensions() == 4) {
                MNN_PRINT("Dimensions: 4, W,H,C,B: %d X %d X %d X %d, OP name %s : %d\n",
                        tensor->width(), tensor->height(), tensor->channel(), tensor->batch(), opName.c_str(), i);
            } else {
                std::ostringstream oss;
                for (int i = 0; i < tensor->dimensions(); i++) {
                    oss << (i ? " X " : "") << tensor->length(i);
                }

                MNN_PRINT("Dimensions: %d, %s, OP name %s : %d\n", tensor->dimensions(), oss.str().c_str(), opName.c_str(), i);
            }

            outputFileName << "output/" << opCopyName << "_" << i;
            dumpTensor2File(expectTensor, outputFileName.str().c_str());
            delete expectTensor;
        }
        return true;
    };
    Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        MNN_PRINT("Usage: ./pictureRecognition_batch.out model.mnn imagedir/ groundtruth.txt quantized.json batchsize total_imgs \n");
        return 0;
    }

    rapidjson::Document document;
    {
        auto configPath = argv[4];
        FUNC_PRINT_ALL(configPath, s);
        std::ifstream fileNames(configPath);
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
    ImageProcess::Config _imageProcessConfig;
    _imageProcessConfig.filterType = BILINEAR;
    _imageProcessConfig.sourceFormat = RGBA;
    _imageProcessConfig.destFormat   = BGR;

    {
        if (picObj.HasMember("format")) {
            auto format = picObj["format"].GetString();
            static std::map<std::string, ImageFormat> formatMap{{"BGR", BGR}, {"RGB", RGB}, {"GRAY", GRAY}, {"RGBA", RGBA}, {"BGRA", BGRA}};
            if (formatMap.find(format) != formatMap.end()) {
                _imageProcessConfig.destFormat = formatMap.find(format)->second;
            }
        }
    }

    _imageProcessConfig.sourceFormat = RGBA;
    int width = 224;
    int height = 224;
    {
        if (picObj.HasMember("width")) {
            width = picObj["width"].GetInt();
        }
        if (picObj.HasMember("height")) {
            height = picObj["height"].GetInt();
        }
        if (picObj.HasMember("mean")) {
            auto mean = picObj["mean"].GetArray();
            int cur   = 0;
            for (auto iter = mean.begin(); iter != mean.end(); iter++) {
                _imageProcessConfig.mean[cur++] = iter->GetFloat();
            }
        }
        if (picObj.HasMember("normal")) {
            auto normal = picObj["normal"].GetArray();
            int cur     = 0;
            for (auto iter = normal.begin(); iter != normal.end(); iter++) {
                _imageProcessConfig.normal[cur++] = iter->GetFloat();
            }
        }
    }

    // Load module with Runtime
    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_AUTO;
    if (picObj.HasMember("CPU")) {
        if (picObj["CPU"].GetBool()) {
            sConfig.type = MNN_FORWARD_CPU;
        }
    }

    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }

    if (false) {
        _initDebug();
        rtmgr->setMode(Interpreter::Session_Debug);
    }
    
    // Give cache full path which must be Readable and writable
    rtmgr->setCache(".cachefile");
    
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));

    string pathToImageTxt(argv[3]);
    string pathToImages(argv[2]);
    std::vector<std::pair<std::string, std::vector<int> > > allTxtLines;
    std::ifstream txtFile(argv[3]);
    if (!txtFile.is_open()) {
        MNN_PRINT("%s: file not found\n", argv[2]);
        MNN_ASSERT(false);
    }
    std::string line;
    while (getline(txtFile, line)) {
        vector<string> splitStr;
        splitStr = split(line, " ");
        if (splitStr.size() != 2) {
            MNN_PRINT("%s: file format error\n", pathToImageTxt.c_str());
            MNN_ASSERT(false);
        }
        std::pair<std::string, std::vector<int> > dataPair;
        dataPair.first = pathToImages + splitStr[0];
        vector<string> labels;
        labels = split(splitStr[1], ",");
        for (int i = 0; i < labels.size(); i++) {
            dataPair.second.emplace_back(atoi(labels[i].c_str()));
        }
        allTxtLines.emplace_back(dataPair);
    }
    txtFile.close();


    // Create Input
    // int batchSize = allTxtLines.size();
    int batchSize = 10;
    int total_images = 50;
    float correct = 0;
    if (argc > 5)
        batchSize = atoi(argv[5]);
    if (argc > 6)
        total_images = atoi(argv[6]);
    int iterations = total_images / batchSize;

    for (int iter = 0; iter < iterations; iter++) {
        
        vector<int> labels;
        auto input = MNN::Express::_Input({batchSize, 3, width, height}, MNN::Express::NC4HW4);
        for (int batch = 0; batch < batchSize; ++batch) {
            int size_w   = width;
            int size_h   = height;
            int bpp      = 3;

            int batchIndex = iter * batchSize + batch;

            auto inputPatch = allTxtLines[batchIndex].first.c_str();
            int inputWidth, inputHeight, channel;
            labels.push_back(allTxtLines[batchIndex].second[0]);
            auto inputImage = stbi_load(inputPatch, &inputWidth, &inputHeight, &channel, 4);
            if (nullptr == inputImage) {
                MNN_ERROR("Can't open %s\n", inputPatch);
                return 0;
            }
            MNN_PRINT("origin size: %d, %d -> %d, %d\n", inputWidth, inputHeight, width, height);
            Matrix trans;
            // Set transform, from dst scale to src, the ways below are both ok
            trans.setScale((float)(inputWidth-1) / (size_w-1), (float)(inputHeight-1) / (size_h-1));
            std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(_imageProcessConfig));
            pretreat->setMatrix(trans);
            // for NC4HW4, UP_DIV(3, 4) * 4 = 4
            pretreat->convert((uint8_t*)inputImage, inputWidth, inputHeight, 0, input->writeMap<float>() + batch * 4 * width * height, width, height, 4, 0,  halide_type_of<float>());
            stbi_image_free(inputImage);
        }
        auto outputs = net->onForward({input});
        auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
        output = MNN::Express::_Reshape(output, {0, -1});
        int topK = 1;
        auto topKV = MNN::Express::_TopKV2(output, MNN::Express::_Scalar<int>(topK));
        auto value = topKV[0];
        auto indices = topKV[1];
        auto valuePtr = topKV[0]->readMap<float>();
        auto indicesPtr = topKV[1]->readMap<int>();
        /*
        for (int batch = 0; batch < batchSize; ++batch) {
            int batchIndex = iter * batchSize + batch;
            MNN_PRINT("For Input: %s, label: %d \n", allTxtLines[batchIndex].first.c_str(), allTxtLines[batchIndex].second[0]);
            for (int i=0; i<topK; ++i) {
                MNN_PRINT("%d, %f\n", indicesPtr[batch * topK + i], valuePtr[batch * topK + i]);
            }
        }*/

        indices = MNN::Express::_Squeeze(indices);
        auto label = MNN::Express::_Const(labels.data(), {int(labels.size())}, MNN::Express::NCHW, halide_type_of<int>());
        auto accu       =MNN::Express::_Cast<int32_t>(MNN::Express::_Equal(MNN::Express::_Cast<int>(indices), label).sum({}));
        correct += accu->readMap<int32_t>()[0];
    }

    MNN_PRINT("batchsize: %d, total: %d, acc: %f %% \n", batchSize, iterations * batchSize, correct / (iterations * batchSize) * 100);

    rtmgr->updateCache();

    return 0;

}