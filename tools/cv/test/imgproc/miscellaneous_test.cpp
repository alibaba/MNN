//
//  miscellaneous_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "test_env.hpp"

#include "cv/imgcodecs.hpp"

#ifdef MNN_MISCELLANEOUS_TEST

static Env<uint8_t> testEnv(img_name, false);

// adaptiveThreshold
TEST(adaptiveThreshold, binary) {
    cv::adaptiveThreshold(testEnv.cvSrcG, testEnv.cvDst, 50, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 2);
    testEnv.mnnDst = adaptiveThreshold(testEnv.mnnSrcG, 50, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 5, 2);
    // EXPECT_TRUE(testEnv.equal());
}

TEST(adaptiveThreshold, binary_inv) {
    cv::adaptiveThreshold(testEnv.cvSrcG, testEnv.cvDst, 50, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 5, 2);
    testEnv.mnnDst = adaptiveThreshold(testEnv.mnnSrcG, 50, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 2);
    // cv::imwrite("cv_res.jpg", testEnv.cvDst);
    // imwrite("mnn_res.jpg", testEnv.mnnDst);
    // EXPECT_TRUE(testEnv.equal());
}

// blendLinear
TEST(blendLinear, 0_6_0_7) {
    int h = testEnv.cvSrc.rows;
    int w = testEnv.cvSrc.cols;
    int weightSize = h * w;
    std::vector<float> weight1(weightSize, 0.6), weight2(weightSize, 0.7);
    cv::Mat cvWeight1 = cv::Mat(h, w, CV_32FC1);
    cv::Mat cvWeight2 = cv::Mat(h, w, CV_32FC1);
    memcpy(cvWeight1.data, weight1.data(), weight1.size() * sizeof(float));
    memcpy(cvWeight2.data, weight2.data(), weight2.size() * sizeof(float));
    VARP mnnWeight1 = _Const(weight1.data(), {h, w, 1}, NHWC, halide_type_of<float>());
    VARP mnnWeight2 = _Const(weight2.data(), {h, w, 1}, NHWC, halide_type_of<float>());
    cv::blendLinear(testEnv.cvSrc, testEnv.cvSrc, cvWeight1, cvWeight2, testEnv.cvDst);
    testEnv.mnnDst = blendLinear(testEnv.mnnSrc, testEnv.mnnSrc, mnnWeight1, mnnWeight2);
    EXPECT_TRUE(testEnv.equal());
}

TEST(blendLinear, 12_0_5) {
    int h = testEnv.cvSrc.rows;
    int w = testEnv.cvSrc.cols;
    int weightSize = h * w;
    std::vector<float> weight1(weightSize, 12), weight2(weightSize, 0.5);
    cv::Mat cvWeight1 = cv::Mat(h, w, CV_32FC1);
    cv::Mat cvWeight2 = cv::Mat(h, w, CV_32FC1);
    memcpy(cvWeight1.data, weight1.data(), weight1.size() * sizeof(float));
    memcpy(cvWeight2.data, weight2.data(), weight2.size() * sizeof(float));
    VARP mnnWeight1 = _Const(weight1.data(), {h, w, 1}, NHWC, halide_type_of<float>());
    VARP mnnWeight2 = _Const(weight2.data(), {h, w, 1}, NHWC, halide_type_of<float>());
    cv::blendLinear(testEnv.cvSrc, testEnv.cvSrc, cvWeight1, cvWeight2, testEnv.cvDst);
    testEnv.mnnDst = blendLinear(testEnv.mnnSrc, testEnv.mnnSrc, mnnWeight1, mnnWeight2);
    EXPECT_TRUE(testEnv.equal());
}

// threshold
TEST(threshold, binary) {
    cv::threshold(testEnv.cvSrc, testEnv.cvDst, 50, 20, cv::THRESH_BINARY);
    testEnv.mnnDst = threshold(testEnv.mnnSrc, 50, 20, THRESH_BINARY);
    EXPECT_TRUE(testEnv.equal());
}

TEST(threshold, binary_inv) {
    cv::threshold(testEnv.cvSrc, testEnv.cvDst, 50, 20, cv::THRESH_BINARY_INV);
    testEnv.mnnDst = threshold(testEnv.mnnSrc, 50, 20, THRESH_BINARY_INV);
    EXPECT_TRUE(testEnv.equal());
}

TEST(threshold, trunc) {
    cv::threshold(testEnv.cvSrc, testEnv.cvDst, 50, 20, cv::THRESH_TRUNC);
    testEnv.mnnDst = threshold(testEnv.mnnSrc, 50, 20, THRESH_TRUNC);
    EXPECT_TRUE(testEnv.equal());
}

TEST(threshold, tozero_inv) {
    cv::threshold(testEnv.cvSrc, testEnv.cvDst, 50, 20, cv::THRESH_TOZERO_INV);
    testEnv.mnnDst = threshold(testEnv.mnnSrc, 50, 20, THRESH_TOZERO_INV);
    EXPECT_TRUE(testEnv.equal());
}

TEST(threshold, tozero) {
    cv::threshold(testEnv.cvSrc, testEnv.cvDst, 50, 20, cv::THRESH_TOZERO);
    testEnv.mnnDst = threshold(testEnv.mnnSrc, 50, 20, THRESH_TOZERO);
    EXPECT_TRUE(testEnv.equal());
}

#endif
