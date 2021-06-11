//
//  CropAndResizeTest.cpp
//  MNNTests
//
//  Created by MNN on 2020/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "MNNTestSuite.h"
#include "TestUtils.h"

using namespace MNN::Express;
class CropAndResizeTest : public MNNTestCase {
public:
    virtual ~CropAndResizeTest() = default;
    virtual bool run(int precision) {
        const int batch = 2, inputHeight = 16, inputWidth = 16, depth = 4, boxNum = 2;
        auto img       = _Input({batch, inputHeight, inputWidth, depth}, NHWC);
        auto boxes     = _Input({boxNum, 4}, NHWC);
        auto box_ind   = _Input({boxNum}, NHWC, halide_type_of<uint32_t>());
        auto crop_size = _Input({2}, NHWC, halide_type_of<uint32_t>());
        auto imgPtr    = img->writeMap<float>();
        for (int i = 0; i < batch * inputHeight * inputWidth * depth; i++) {
            imgPtr[i] = static_cast<float>((i % 255) / 255.0f);
        }
        const float box_data[] = {
            0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.6, 0.7,
        };
        memcpy(boxes->writeMap<float>(), box_data, boxNum * 4 * sizeof(float));
        box_ind->writeMap<uint32_t>()[0]        = 1;
        box_ind->writeMap<uint32_t>()[1]        = 0;
        crop_size->writeMap<uint32_t>()[0]      = 4;
        crop_size->writeMap<uint32_t>()[1]      = 4;
        auto output                             = _CropAndResize(img, boxes, box_ind, crop_size, BILINEAR);
        const std::vector<float> expectedOutput = {
            0.8392, 0.8431, 0.8471, 0.8510, 0.8549, 0.8588, 0.8627, 0.8667, 0.8706, 0.8745, 0.8784, 0.8824, 0.8863,
            0.8902, 0.8941, 0.8980,

            0.0902, 0.0941, 0.0980, 0.1020, 0.1059, 0.1098, 0.1137, 0.1176, 0.1216, 0.1255, 0.1294, 0.1333, 0.1373,
            0.1412, 0.1451, 0.1490,

            0.3412, 0.3451, 0.3490, 0.3529, 0.3569, 0.3608, 0.3647, 0.3686, 0.3725, 0.3765, 0.3804, 0.3843, 0.3882,
            0.3922, 0.3961, 0.4000,

            0.5922, 0.5961, 0.6000, 0.6039, 0.6078, 0.6118, 0.6157, 0.6196, 0.6235, 0.6275, 0.6314, 0.6353, 0.6392,
            0.6431, 0.6471, 0.6510,

            0.2235, 0.2275, 0.2314, 0.2353, 0.2471, 0.2510, 0.2549, 0.2588, 0.2706, 0.2745, 0.2784, 0.2824, 0.2941,
            0.2980, 0.3020, 0.3059, 0.6000, 0.6039, 0.6078, 0.6118, 0.6235, 0.6275, 0.6314, 0.6353, 0.6471, 0.6510,
            0.6549, 0.6588, 0.6706, 0.6745, 0.6784, 0.6824, 0.4765, 0.4804, 0.4843, 0.4882, 0.5000, 0.5039, 0.5078,
            0.5118, 0.5235, 0.5275, 0.5314, 0.5353, 0.5471, 0.5510, 0.5549, 0.5588, 0.3529, 0.3569, 0.3608, 0.3647,
            0.3765, 0.3804, 0.3843, 0.3882, 0.4000, 0.4039, 0.4078, 0.4118, 0.4235, 0.4275, 0.4314, 0.4353};
        auto gotOutput = output->readMap<float>();
        if (!checkVector<float>(gotOutput, expectedOutput.data(), expectedOutput.size(), 0.01)) {
            MNN_ERROR("CropAndResizeTest test failed!\n");
            return false;
        }
        return true;
    }
};
MNNTestSuiteRegister(CropAndResizeTest, "op/CropAndResize");
