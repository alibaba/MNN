//
//  histograms_test.cpp
//  MNN
//
//  Created by MNN on 2022/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "test_env.hpp"

#ifdef MNN_HISTOGRAMS_TEST

static Env<uint8_t> testEnv(img_name, false);

// histograms
TEST(histograms, channel_0) {
    std::vector<cv::Mat> images {testEnv.cvSrc};
    std::vector<int> histSize {256};
    std::vector<int> channels {0};
    std::vector<float> ranges {0., 256.};
    cv::calcHist(images, channels, cv::Mat(), testEnv.cvDst, histSize, ranges);
    testEnv.mnnDst = calcHist({testEnv.mnnSrc}, channels, nullptr, histSize, ranges);
    bool eq = _equal<float, float>(testEnv.cvDst, testEnv.mnnDst);
    ASSERT_TRUE(eq);
}

TEST(histograms, channel_2) {
    std::vector<cv::Mat> images {testEnv.cvSrc};
    std::vector<int> histSize {256};
    std::vector<int> channels {2};
    std::vector<float> ranges {0., 256.};
    cv::calcHist(images, channels, cv::Mat(), testEnv.cvDst, histSize, ranges);
    testEnv.mnnDst = calcHist({testEnv.mnnSrc}, channels, nullptr, histSize, ranges);
    bool eq = _equal<float, float>(testEnv.cvDst, testEnv.mnnDst);
    ASSERT_TRUE(eq);
}

#endif
