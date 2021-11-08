//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./segment.out model.mnn input.jpg output.jpg\n");
        return 0;
    }
    std::shared_ptr<Interpreter> net;
    net.reset(Interpreter::createFromFile(argv[1]));
    if (net == nullptr) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    ScheduleConfig config;
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, nullptr);
    auto shape = input->shape();
    if (shape[0] != 1) {
        shape[0] = 1;
        net->resizeTensor(input, shape);
        net->resizeSession(session);
    }
    {
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        bpp = shape[1];
        size_h = shape[2];
        size_w = shape[3];
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
        config.filterType = CV::BILINEAR;
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
    // Run model
    net->runSession(session);

    // Post treat by MNN-Express
    {
        /* Create VARP by tensor Begin*/
        auto outputTensor = net->getSessionOutput(session, nullptr);
        // First Create a Expr, then create Variable by the 0 index of expr
        auto output = Variable::create(Expr::create(outputTensor));
        if (nullptr == output->getInfo()) {
            MNN_ERROR("Alloc memory or compute size error\n");
            return 0;
        }
        /* Create VARP by tensor End*/

        // Turn dataFormat to NHWC for easy to run TopKV2
        output = _Convert(output, NHWC);
        auto width = output->getInfo()->dim[2];
        auto height = output->getInfo()->dim[1];
        auto channel = output->getInfo()->dim[3];
        MNN_PRINT("output w = %d, h=%d\n", width, height);

        const int humanIndex = 15;
        output = _Reshape(output, {-1, channel});
        auto kv = _TopKV2(output, _Scalar<int>(1));
        // Use indice in TopKV2's C axis
        auto index = kv[1];
        // If is human, set 255, else set 0
        auto mask = _Select(_Equal(index, _Scalar<int>(humanIndex)), _Scalar<int>(255), _Scalar<int>(0));

        //If need faster, use this code
        //auto mask = _Equal(index, _Scalar<int>(humanIndex)) * _Scalar<int>(255);

        mask = _Cast<uint8_t>(mask);
        stbi_write_png(argv[3], width, height, 1, mask->readMap<uint8_t>(), width);
    }
    return 0;
}
