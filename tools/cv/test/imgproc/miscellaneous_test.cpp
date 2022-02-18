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

#ifdef MNN_MISCELLANEOUS_TEST

static Env<float> testEnv("img_name", true);

// blendLinear
TEST(blendLinear, 0_6_0_7) {
    int weightSize = testEnv.cvSrc.rows * testEnv.cvSrc.cols;
    std::vector<float> weight1(weightSize, 0.6), weight2(weightSize, 0.7);
    cv::Mat cvWeight1 = cv::Mat(testEnv.cvSrc.rows, testEnv.cvSrc.cols, CV_32FC1);
    cv::Mat cvWeight2 = cv::Mat(testEnv.cvSrc.rows, testEnv.cvSrc.cols, CV_32FC1);
    memcpy(cvWeight1.data, weight1.data(), weight1.size() * sizeof(float));
    memcpy(cvWeight2.data, weight2.data(), weight2.size() * sizeof(float));
    VARP mnnWeight1, mnnWeight2;
    testEnv.cv2mnn(cvWeight1, mnnWeight1);
    testEnv.cv2mnn(cvWeight2, mnnWeight2);
    cv::blendLinear(testEnv.cvSrc, testEnv.cvSrc, cvWeight1, cvWeight2, testEnv.cvDst);
    testEnv.mnnDst = blendLinear(testEnv.mnnSrc, testEnv.mnnSrc, mnnWeight1, mnnWeight2);
    EXPECT_TRUE(testEnv.equal());
}

TEST(blendLinear, 12_0_5) {
    int weightSize = testEnv.cvSrc.rows * testEnv.cvSrc.cols;
    std::vector<float> weight1(weightSize, 12), weight2(weightSize, 0.5);
    cv::Mat cvWeight1 = cv::Mat(testEnv.cvSrc.rows, testEnv.cvSrc.cols, CV_32FC1);
    cv::Mat cvWeight2 = cv::Mat(testEnv.cvSrc.rows, testEnv.cvSrc.cols, CV_32FC1);
    memcpy(cvWeight1.data, weight1.data(), weight1.size() * sizeof(float));
    memcpy(cvWeight2.data, weight2.data(), weight2.size() * sizeof(float));
    VARP mnnWeight1, mnnWeight2;
    testEnv.cv2mnn(cvWeight1, mnnWeight1);
    testEnv.cv2mnn(cvWeight2, mnnWeight2);
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
