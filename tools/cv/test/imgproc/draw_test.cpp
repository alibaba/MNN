//
//  draw_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv/imgcodecs.hpp"
#include "test_env.hpp"

#define MNN_DRAW_TEST
#ifdef MNN_DRAW_TEST

static Env<uint8_t> testEnv(img_name, false);

/*
// arrowedLine
TEST(arrowedLine, basic) {
    cv::arrowedLine(testEnv.cvSrc, {10, 10}, {300, 200}, {0, 0, 255});
    arrowedLine(testEnv.mnnSrc, {10, 10}, {300, 200}, {0, 0, 255});
    // cv::imwrite("cv_line.jpg", testEnv.cvSrc);
    // imwrite("mnn_line.jpg", testEnv.mnnSrc);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}*/

// line
TEST(line, basic) {
    cv::line(testEnv.cvSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    line(testEnv.mnnSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

/*
TEST(line, thickness) {
    cv::line(testEnv.cvSrc, {10, 10}, {20, 20}, {0, 0, 255}, 1);
    line(testEnv.mnnSrc, {10, 10}, {20, 20}, {0, 0, 255}, 1);
    cv::imwrite("cv_line.jpg", testEnv.cvSrc);
    imwrite("mnn_line.jpg", testEnv.mnnSrc);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}*/

// rectangle
TEST(rectangle, basic) {
    cv::rectangle(testEnv.cvSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    rectangle(testEnv.mnnSrc, {10, 10}, {200, 300}, {0, 0, 255}, 1);
    EXPECT_TRUE(testEnv.equal(testEnv.cvSrc, testEnv.mnnSrc));
}

#endif
