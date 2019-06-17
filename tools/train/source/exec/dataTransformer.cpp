//
//  dataTransformer.cpp
//  MNN
//
//  Created by MNN on 2019/05/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <sstream>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "converter/source/IR/MNN_generated.h"
#include "rapidjson/document.h"
using namespace MNN;
using namespace MNN::CV;
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, const char* argv[]) {
    rapidjson::Document document;
    if (argc < 3) {
        MNN_ERROR("Usage: ./dataTransformer.out mobilenet.alinn picpath.json storage.bin\n");
        return 0;
    }
    FUNC_PRINT_ALL(argv[1], s);
    FUNC_PRINT_ALL(argv[2], s);
    FUNC_PRINT_ALL(argv[3], s);
    std::unique_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig scheduleConfig;
    auto session    = net->createSession(scheduleConfig);
    auto dataTensor = net->getSessionInput(session, nullptr);
    auto probTensor = net->getSessionOutput(session, nullptr);

    {
        std::ifstream fileNames(argv[2]);
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
    }
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> result;

    auto pathArray = picObj["path"].GetArray();
    for (auto iter = pathArray.begin(); iter != pathArray.end(); iter++) {
        auto path = iter->GetString();
        // FUNC_PRINT_ALL(path, s);
        int width, height, channel;
        auto inputImage = stbi_load(path, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Invalid Path: %s\n", path);
            continue;
        }
        Matrix m;
        m.setScale((float)width / dataTensor->width(), (float)height / dataTensor->height());

        process->setMatrix(m);
        process->convert(inputImage, width, height, 0, dataTensor);
        std::shared_ptr<Tensor> userTensor(new Tensor(dataTensor));
        dataTensor->copyToHostTensor(userTensor.get());

        net->runSession(session);

        std::shared_ptr<Tensor> probUserTensor(new Tensor(probTensor));
        probTensor->copyToHostTensor(probUserTensor.get());

        result.emplace_back(std::make_pair(userTensor, probUserTensor));
        stbi_image_free(inputImage);
    }
    {
        std::unique_ptr<NetT> data(new NetT);
        data->tensorName = {net->getSessionInputAll(session).begin()->first,
                            net->getSessionOutputAll(session).begin()->first + "_Compare"};
        {
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->type          = OpType_Const;
            newOp->name          = data->tensorName[0];
            newOp->outputIndexes = {0};
            newOp->main.type     = OpParameter_Blob;
            auto blobT           = new BlobT;
            blobT->dims      = {(int)result.size(), dataTensor->channel(), dataTensor->height(), dataTensor->width()};
            size_t totalSize = 1;
            for (int i = 0; i < blobT->dims.size(); ++i) {
                totalSize *= blobT->dims[i];
            }
            blobT->float32s.resize(totalSize);
            switch (dataTensor->getDimensionType()) {
                case MNN::Tensor::CAFFE:
                    blobT->dataFormat = MNN_DATA_FORMAT_NCHW;
                    break;
                case MNN::Tensor::TENSORFLOW:
                    blobT->dataFormat = MNN_DATA_FORMAT_NHWC;
                    break;
                default:
                    break;
            }
            for (int i = 0; i < result.size(); ++i) {
                auto tensor = result[i].first.get();
                auto dst    = blobT->float32s.data() + i * tensor->elementSize();
                auto src    = tensor->host<float>();
                ::memcpy(dst, src, tensor->size());
            }
            newOp->main.value = blobT;
            data->oplists.emplace_back(std::move(newOp));
        }
        {
            std::unique_ptr<OpT> newOp(new OpT);
            newOp->type          = OpType_Const;
            newOp->name          = data->tensorName[1];
            newOp->outputIndexes = {1};
            newOp->main.type     = OpParameter_Blob;
            auto blobT           = new BlobT;
            for (int i = 0; i < probTensor->dimensions(); ++i) {
                blobT->dims.emplace_back(probTensor->length(i));
            }
            blobT->dims[0]   = result.size();
            size_t totalSize = 1;
            for (int i = 0; i < blobT->dims.size(); ++i) {
                totalSize *= blobT->dims[i];
            }
            switch (probTensor->getDimensionType()) {
                case MNN::Tensor::CAFFE:
                    blobT->dataFormat = MNN_DATA_FORMAT_NCHW;
                    break;
                case MNN::Tensor::TENSORFLOW:
                    blobT->dataFormat = MNN_DATA_FORMAT_NHWC;
                    break;
                default:
                    break;
            }
            blobT->float32s.resize(totalSize);
            for (int i = 0; i < result.size(); ++i) {
                auto tensor = result[i].second.get();
                auto dst    = blobT->float32s.data() + i * tensor->elementSize();
                auto src    = tensor->host<float>();
                ::memcpy(dst, src, tensor->size());
            }
            newOp->main.value = blobT;
            data->oplists.emplace_back(std::move(newOp));
        }
        flatbuffers::FlatBufferBuilder builder(1024);
        auto offset = Net::Pack(builder, data.get());
        builder.Finish(offset);
        std::ofstream os(argv[3]);
        os.write((const char*)builder.GetBufferPointer(), builder.GetSize());
    }

    return 0;
}
