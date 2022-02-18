//
//  color_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "test_env.hpp"

#ifdef MNN_TEST_COLOR

static Env<unsigned char> testEnv(img_name, false);

// cvtColor
#define TEST_CV_COLOR(code, src) \
TEST(cvtColor, COLOR_##code) {\
    cv::cvtColor(testEnv.cvSrc##src, testEnv.cvDst, cv::COLOR_##code);\
    testEnv.mnnDst = cvtColor(testEnv.mnnSrc##src, COLOR_##code);\
    EXPECT_TRUE(testEnv.equal());\
}

// RGB -> *
TEST_CV_COLOR(RGB2BGR,)
TEST_CV_COLOR(RGB2GRAY,)
TEST_CV_COLOR(RGB2RGBA,)
TEST_CV_COLOR(RGB2YCrCb,)
TEST_CV_COLOR(RGB2YUV,)
TEST_CV_COLOR(RGB2XYZ,)
TEST_CV_COLOR(RGB2HSV,)
TEST_CV_COLOR(RGB2HSV_FULL,)
TEST_CV_COLOR(RGB2BGR555,)
TEST_CV_COLOR(RGB2BGR565,)

// BGR -> *
TEST_CV_COLOR(BGR2RGB,)
TEST_CV_COLOR(BGR2GRAY,)
TEST_CV_COLOR(BGR2BGRA,)
TEST_CV_COLOR(BGR2YCrCb,)
TEST_CV_COLOR(BGR2YUV,)
TEST_CV_COLOR(BGR2XYZ,)
TEST_CV_COLOR(BGR2HSV,)
TEST_CV_COLOR(BGR2HSV_FULL,)
TEST_CV_COLOR(BGR2BGR555,)
TEST_CV_COLOR(BGR2BGR565,)

// RGBA -> *
TEST_CV_COLOR(RGBA2BGRA, A)
TEST_CV_COLOR(RGBA2BGR, A)
TEST_CV_COLOR(RGBA2RGB, A)
TEST_CV_COLOR(RGBA2GRAY, A)

// BGRA -> *
TEST_CV_COLOR(BGRA2RGBA, A)
TEST_CV_COLOR(BGRA2BGR, A)
TEST_CV_COLOR(BGRA2RGB, A)
TEST_CV_COLOR(BGRA2GRAY, A)

// GRAY -> *
TEST_CV_COLOR(GRAY2RGBA, G)
TEST_CV_COLOR(GRAY2BGRA, G)
TEST_CV_COLOR(GRAY2BGR, G)
TEST_CV_COLOR(GRAY2RGB, G)
/*
// YUV_I420 -> *
TEST_CV_COLOR(YUV2RGB_I420, Y)
TEST_CV_COLOR(YUV2BGR_I420, Y)
TEST_CV_COLOR(YUV2RGBA_I420, Y)
TEST_CV_COLOR(YUV2BGRA_I420, Y)
TEST_CV_COLOR(YUV2GRAY_I420, Y)
*/

#if 0
// mnn's YUV -> RGB is different from opencv, but it's right
// cvtColorTwoPlane
#define TEST_CV_TWO_COLOR(code) \
TEST(cvtColorTwoPlane, COLOR_##code) {\
    std::vector<uchar> y_data(640 * 480, 64);\
    std::vector<uchar> uv_data(640 * 240, 64);\
    cv::Mat cvY(480, 640, CV_8UC1, y_data.data(), 640);\
    cv::Mat cvUV(240, 320, CV_8UC2, uv_data.data());\
    VARP mnnY, mnnUV;\
    testEnv.cv2mnn(cvY, mnnY);\
    testEnv.cv2mnn(cvUV, mnnUV);\
    cv::cvtColorTwoPlane(cvY, cvUV, testEnv.cvDst, cv::COLOR_##code);\
    testEnv.mnnDst = cvtColorTwoPlane(mnnY, mnnUV, COLOR_##code);\
    EXPECT_TRUE(testEnv.equal());\
}
TEST_CV_TWO_COLOR(YUV2RGB_NV21)
TEST_CV_TWO_COLOR(YUV2BGR_NV21)
TEST_CV_TWO_COLOR(YUV2RGBA_NV21)
TEST_CV_TWO_COLOR(YUV2BGRA_NV21)

TEST_CV_TWO_COLOR(YUV2RGB_NV12)
TEST_CV_TWO_COLOR(YUV2BGR_NV12)
TEST_CV_TWO_COLOR(YUV2RGBA_NV12)
TEST_CV_TWO_COLOR(YUV2BGRA_NV12)
#endif

#endif
