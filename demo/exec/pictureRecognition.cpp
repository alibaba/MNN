//
//  pictureTest.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#include "FreeImage.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./pictureTest.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    auto input  = net->getSessionInput(session, NULL);
    auto output = net->getSessionOutput(session, NULL);
    std::vector<std::string> words;
    if (argc >= 4) {
        std::ifstream inputOs(argv[3]);
        std::string line;
        while (std::getline(inputOs, line)) {
            words.emplace_back(line);
        }
    }
    {
        auto dims    = input->shape();
        int inputDim = 0;
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        bpp          = input->channel();
        size_h       = input->height();
        size_w       = input->width();
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch     = argv[2];
        FREE_IMAGE_FORMAT f = FreeImage_GetFileType(inputPatch);
        FIBITMAP* bitmap    = FreeImage_Load(f, inputPatch);
        MNN_ASSERT(NULL != bitmap);
        auto newBitmap = FreeImage_ConvertTo32Bits(bitmap);
        FreeImage_Unload(bitmap);
        auto width  = FreeImage_GetWidth(newBitmap);
        auto height = FreeImage_GetHeight(newBitmap);
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Dst -> [0, 1]
        trans.postScale(1.0 / size_w, 1.0 / size_h);
        // Flip Y
        trans.postScale(1.0, -1.0, 0.0, 0.5);
        //[0, 1] -> Src
        trans.postScale(width, height);
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3]     = {103.94f, 116.78f, 123.68f};
        float normals[3]  = {0.017f, 0.017f, 0.017f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)FreeImage_GetScanLine(newBitmap, 0), width, height, 0, input);
        FreeImage_Save(FIF_PNG, newBitmap, argv[3], PNG_DEFAULT);
        FreeImage_Unload(newBitmap);
    }
    net->runSession(session);
    {
        MNN_PRINT("output size:%d\n", output->elementSize());
        auto type = output->getType();

        auto size = output->elementSize();
        std::vector<std::pair<int, float>> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = output->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = std::make_pair(i, values[i]);
            }
        }
        if (type.code == halide_type_uint && type.bytes() == 1) {
            auto values = output->host<uint8_t>();
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
