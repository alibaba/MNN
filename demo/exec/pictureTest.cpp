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
#include <sstream>
#include <vector>
#include "AutoTime.hpp"
#include "FreeImage.h"
using namespace MNN;
using namespace MNN::CV;

void _testYUV() {
    std::ifstream yuvStream("yuvbuffer");
    std::ostringstream yuvStreamOs;
    yuvStreamOs << yuvStream.rdbuf();

    auto yuvBuffer = yuvStreamOs.str();
    printf("yuvbuffer size: %lu\n", yuvBuffer.size());

    int size_w     = 1280;
    int size_h     = 960;
    int dstW       = 456;
    int dstH       = 400;
    int bpp        = 3;
    auto rgbBitmap = FreeImage_Allocate(dstW, dstH, bpp * 8);
    ImageProcess::Config config;
    config.filterType   = BILINEAR;
    config.sourceFormat = YUV_NV21;
    config.destFormat   = RGB;
    std::shared_ptr<ImageProcess> process(ImageProcess::create(config));
    // //trans.setIdentity();
    Matrix trans2;
    // Dst -> [0, 1]
    trans2.postScale(1.0 / dstW, 1.0 / dstH);
    //[0, 1] -> Src
    trans2.postScale(size_w, size_h);

    process->setMatrix(trans2);

    auto tempTensor = ImageProcess::createImageTensor<uint8_t>(dstW, dstH, bpp, FreeImage_GetScanLine(rgbBitmap, 0));
    process->convert((uint8_t*)yuvBuffer.c_str(), size_w, size_h, 0, tempTensor);
    FreeImage_Save(FIF_PNG, rgbBitmap, "yuv.png", PNG_DEFAULT);
    FreeImage_Unload(rgbBitmap);
    delete tempTensor;
}

int main(int argc, const char* argv[]) {
    //_testYUV();
    if (argc < 4) {
        printf("Usage: ./pictureTest.out model.mnn input.jpg output.jpg\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);

    auto input  = net->getSessionInput(session, NULL);
    auto output = net->getSessionOutput(session, NULL);
    {
        auto dims    = input->shape();
        int inputDim = 0;
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        bpp          = dims[1];
        size_h       = dims[2];
        size_w       = dims[3];
        if (input->getDimensionType() == MNN::Tensor::TENSORFLOW) {
            bpp    = dims[3];
            size_h = dims[1];
            size_w = dims[2];
        }
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        printf("input_%d.txt: w:%d , h:%d, bpp: %d\n", inputDim, size_w, size_h, bpp);

        auto inputPatch     = argv[2];
        FREE_IMAGE_FORMAT f = FreeImage_GetFileType(inputPatch);
        FIBITMAP* bitmap    = FreeImage_Load(f, inputPatch);
        MNN_ASSERT(NULL != bitmap);
        auto newBitmap = FreeImage_ConvertTo32Bits(bitmap);
        FreeImage_Unload(bitmap);
        auto width  = FreeImage_GetWidth(newBitmap);
        auto height = FreeImage_GetHeight(newBitmap);
        printf("origin size: %d, %d\n", width, height);
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
        if (false) {
            std::ofstream outputOs("input_pictureTest.txt");
            int size2      = input->elementSize();
            auto inputData = input->host<float>();
            for (int v = 0; v < size2; ++v) {
                outputOs << inputData[v] << "\n";
            }
        }

        FreeImage_Save(FIF_PNG, newBitmap, argv[3], PNG_DEFAULT);
        FreeImage_Unload(newBitmap);
    }
    net->runSession(session);

    if (false) {
        std::ofstream outputOs("output_pictureTest.txt");
        int size2      = output->elementSize();
        auto inputData = output->host<float>();
        for (int v = 0; v < size2; ++v) {
            outputOs << inputData[v] << "\n";
        }
    }
    {
        printf("output size:%d\n", output->elementSize());
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
        for (int i = 0; i < length; ++i) {
            printf("%d, %f\n", tempValues[i].first, tempValues[i].second);
        }
    }
    return 0;
}
