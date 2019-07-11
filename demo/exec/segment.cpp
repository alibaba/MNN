//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./segment.out model.mnn input.jpg output.jpg\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    
    auto input = net->getSessionInput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto output = net->getSessionOutput(session, NULL);
    
    {
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
        
        auto inputPatch = argv[2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Set scale, from dst scale to src
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        //        float mean[3]     = {103.94f, 116.78f, 123.68f};
        //        float normals[3] = {0.017f, 0.017f, 0.017f};
        float mean[3]     = {127.5f, 127.5f, 127.5f};
        float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = RGB;
        
        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input);
        stbi_image_free(inputImage);
    }
    net->runSession(session);
    {
        std::shared_ptr<Tensor> outputUser(new Tensor(output, Tensor::TENSORFLOW));
        MNN_PRINT("output size:%d x %d x %d\n", outputUser->width(), outputUser->height(), outputUser->channel());
        output->copyToHostTensor(outputUser.get());
        
        auto width = outputUser->width();
        auto height = outputUser->height();
        auto channel = outputUser->channel();
        std::shared_ptr<Tensor> wrapTensor(ImageProcess::createImageTensor<uint8_t>(outputUser->width(), outputUser->height(), 4, nullptr));
        for (int y = 0; y < height; ++y) {
            auto rgbaY = wrapTensor->host<uint8_t>() + 4 * y * width;
            auto sourceY = outputUser->host<float>() + y * width * channel;
            for (int x=0; x<width; ++x) {
                auto sourceX = sourceY + channel * x;
                int index = 0;
                float maxValue = sourceX[0];
                auto rgba = rgbaY + 4 * x;
                for (int c=1; c<channel; ++c) {
                    if (sourceX[c] > maxValue) {
                        index = c;
                        maxValue = sourceX[c];
                    }
                }
                rgba[0] = 255;
                rgba[2] = 0;
                rgba[1] = 0;
                rgba[3] = 255;
                if (15 == index) {
                    rgba[2] = 255;
                    rgba[3] = 0;
                }
            }
        }
        stbi_write_png(argv[3], width, height, 4, wrapTensor->host<uint8_t>(), 4 * width);
    }
    return 0;
}
