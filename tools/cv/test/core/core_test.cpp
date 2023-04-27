//
//  core_test.cpp
//  MNN
//
//  Created by MNN on 2023/04/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <MNN/AutoTime.hpp>
#include "cv/core.hpp"
#include "test_env.hpp"

#ifdef MNN_CORE_TEST

static Env<uint8_t> testEnv(img_name, false);

// solve
TEST(solve, 1x1) {
    float A[4] = { 2 };
    float B[2] = { 4 };
    VARP mnnA = _Const(A, {1, 1});
    VARP mnnB = _Const(B, {1, 1});
    cv::Mat cvA = cv::Mat(1, 1, CV_32F, A);
    cv::Mat cvB = cv::Mat(1, 1, CV_32F, B);
    cv::Mat cvRes;
    auto cvres = cv::solve(cvA, cvB, cvRes);
    auto mnnRes = solve(mnnA, mnnB);
    EXPECT_EQ(cvres, mnnRes.first);
    EXPECT_TRUE(_equal<float>(cvRes, mnnRes.second));
}

TEST(solve, 2x2) {
    float A[4] = { 1, 2, 3, 4 };
    float B[2] = { 5, 11 };
    VARP mnnA = _Const(A, {2, 2});
    VARP mnnB = _Const(B, {2, 1});
    cv::Mat cvA = cv::Mat(2, 2, CV_32F, A);
    cv::Mat cvB = cv::Mat(2, 1, CV_32F, B);
    cv::Mat cvRes;
    auto cvres = cv::solve(cvA, cvB, cvRes);
    auto mnnRes = solve(mnnA, mnnB);
    EXPECT_EQ(cvres, mnnRes.first);
    EXPECT_TRUE(_equal<float>(cvRes, mnnRes.second));
}

TEST(solve, 3x3) {
    float A[9] = { 2, 3, 4, 0, 1, 5, 0, 0, 3 };
    float B[3] = { 1, 2, 3 };
    VARP mnnA = _Const(A, {3, 3});
    VARP mnnB = _Const(B, {3, 1});
    cv::Mat cvA = cv::Mat(3, 3, CV_32F, A);
    cv::Mat cvB = cv::Mat(3, 1, CV_32F, B);
    cv::Mat cvRes;
    auto cvres = cv::solve(cvA, cvB, cvRes);
    auto mnnRes = solve(mnnA, mnnB);
    EXPECT_EQ(cvres, mnnRes.first);
    EXPECT_TRUE(_equal<float>(cvRes, mnnRes.second));
}

TEST(solve, 6x6) {
    float A[64] = {
        1, 2, 3, 4, 5, 6,
        2, 3, 4, 5, 6, 1,
        3, 4, 5, 6, 1, 2,
        4, 5, 6, 1, 2, 3,
        5, 6, 1, 2, 3, 4,
        6, 1, 2, 3, 4, 5
    };
    float B[6] = { 20, 23, 25, 22, 21, 24 };
    VARP mnnA = _Const(A, {6, 6});
    VARP mnnB = _Const(B, {6, 1});
    cv::Mat cvA = cv::Mat(6, 6, CV_32F, A);
    cv::Mat cvB = cv::Mat(6, 1, CV_32F, B);
    cv::Mat cvRes;
    auto cvres = cv::solve(cvA, cvB, cvRes, DECOMP_SVD);
    auto mnnRes = solve(mnnA, mnnB, DECOMP_SVD);
    EXPECT_EQ(cvres, mnnRes.first);
    EXPECT_TRUE(_equal<float>(cvRes, mnnRes.second));
}

TEST(solve, 8x8) {
    float A[64] = {
        2, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 1, 1, 1, 1, 1, 1,
        1, 1, 2, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 1, 1, 1, 1,
        1, 1, 1, 1, 2, 1, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 1, 1, 2
    };
    float B[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    VARP mnnA = _Const(A, {8, 8});
    VARP mnnB = _Const(B, {8, 1});
    cv::Mat cvA = cv::Mat(8, 8, CV_32F, A);
    cv::Mat cvB = cv::Mat(8, 1, CV_32F, B);
    cv::Mat cvRes;
    auto cvres = cv::solve(cvA, cvB, cvRes);
    auto mnnRes = solve(mnnA, mnnB);
    EXPECT_EQ(cvres, mnnRes.first);
    EXPECT_TRUE(_equal<float>(cvRes, mnnRes.second));
}

#endif
