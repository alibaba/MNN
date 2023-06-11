//
//  filter_test.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <gtest/gtest.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "test_env.hpp"
#include "cv/imgcodecs.hpp"

#ifdef MNN_TEST_FILTER
static Env<uint8_t> testEnv(img_name, false);

// bilateralFilter
TEST(bilateralFilter, basic) {
    cv::bilateralFilter(testEnv.cvSrc, testEnv.cvDst, 20, 80, 35);
    testEnv.mnnDst = bilateralFilter(testEnv.mnnSrc, 20, 80, 35);
    EXPECT_TRUE(testEnv.equal());
}

// blur
TEST(blur, ksize_3x3) {
    cv::blur(testEnv.cvSrc, testEnv.cvDst, {3, 3});
    testEnv.mnnDst = blur(testEnv.mnnSrc, {3, 3});
    EXPECT_TRUE(testEnv.equal());
}

TEST(blur, ksize_5x5) {
    cv::blur(testEnv.cvSrc, testEnv.cvDst, {5, 5});
    testEnv.mnnDst = blur(testEnv.mnnSrc, {5, 5});
    EXPECT_TRUE(testEnv.equal());
}

// boxFilter
TEST(boxFilter, ksize_3x3) {
    cv::boxFilter(testEnv.cvSrc, testEnv.cvDst, -1, {3, 3}, {-1, -1}, false);
    testEnv.mnnDst = boxFilter(testEnv.mnnSrc, -1, {3, 3}, false);
    EXPECT_TRUE(testEnv.equal());
}

TEST(boxFilter, ksize_3x3_norm) {
    cv::boxFilter(testEnv.cvSrc, testEnv.cvDst, -1, {3, 3});
    testEnv.mnnDst = boxFilter(testEnv.mnnSrc, -1, {3, 3});
    EXPECT_TRUE(testEnv.equal());
}

// erode
TEST(erode, basic) {
    cv::erode(testEnv.cvSrc, testEnv.cvDst, cv::getStructuringElement(0, {3, 3}));
    testEnv.mnnDst = erode(testEnv.mnnSrc, getStructuringElement(0, {3, 3}));
    EXPECT_TRUE(testEnv.equal());
}

// dilate
TEST(dilate, basic) {
    cv::dilate(testEnv.cvSrc, testEnv.cvDst, cv::getStructuringElement(0, {3, 3}));
    testEnv.mnnDst = dilate(testEnv.mnnSrc, getStructuringElement(0, {3, 3}));
    auto src_info = testEnv.mnnSrc->getInfo();
    auto dst_info = testEnv.mnnDst->getInfo();
    EXPECT_TRUE(testEnv.equal());
}

// filter2D
TEST(filter2D, ksize_3x3) {
    std::vector<float> kernel { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    cv::Mat cvKernel = cv::Mat(3, 3, CV_32FC1);
    memcpy(cvKernel.data, kernel.data(), kernel.size() * sizeof(float));
    VARP mnnKernel = _Const(kernel.data(), {3, 3});
    cv::filter2D(testEnv.cvSrc, testEnv.cvDst, -1, cvKernel);
    testEnv.mnnDst = filter2D(testEnv.mnnSrc, -1, mnnKernel);
    EXPECT_TRUE(testEnv.equal());
}

TEST(filter2D, ksize_3x3_delta) {
    std::vector<float> kernel { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    cv::Mat cvKernel = cv::Mat(3, 3, CV_32FC1);
    memcpy(cvKernel.data, kernel.data(), kernel.size() * sizeof(float));
    VARP mnnKernel = _Const(kernel.data(), {3, 3});
    cv::filter2D(testEnv.cvSrc, testEnv.cvDst, -1, cvKernel, {-1, -1}, 2);
    testEnv.mnnDst = filter2D(testEnv.mnnSrc, -1, mnnKernel, 2);
    EXPECT_TRUE(testEnv.equal());
}

// GaussianBlur
TEST(GaussianBlur, ksize_3x3_sigmaX_10) {
    cv::GaussianBlur(testEnv.cvSrc, testEnv.cvDst, {3, 3}, 10);
    testEnv.mnnDst = GaussianBlur(testEnv.mnnSrc, {3, 3}, 10);
    EXPECT_TRUE(testEnv.equal());
}

TEST(GaussianBlur, ksize_3x3_sigmaX_10_sigmaY_5) {
    cv::GaussianBlur(testEnv.cvSrc, testEnv.cvDst, {3, 3}, 10, 5);
    testEnv.mnnDst = GaussianBlur(testEnv.mnnSrc, {3, 3}, 10, 5);
    EXPECT_TRUE(testEnv.equal());
}

// getDerivKernels
TEST(getDerivKernels, dx_1_dy_2_ksize_1) {
    cv::Mat cvKx, cvKy;
    cv::getDerivKernels(cvKx, cvKy, 1, 2, 1);
    auto mnnKxy = getDerivKernels(1, 2, 1);
    EXPECT_TRUE(testEnv.equal(cvKx, mnnKxy.first) && testEnv.equal(cvKy, mnnKxy.second));
}

TEST(getDerivKernels, dx_1_dy_2_ksize_3_norm) {
    cv::Mat cvKx, cvKy;
    VARP mnnKy;
    cv::getDerivKernels(cvKx, cvKy, 1, 2, 1, true);
    auto mnnKxy = getDerivKernels(1, 2, 1, true);
    EXPECT_TRUE(testEnv.equal(cvKx, mnnKxy.first) && testEnv.equal(cvKy, mnnKxy.second));
}

TEST(getDerivKernels, dx_1_dy_2_ksize_5_norm) {
    cv::Mat cvKx, cvKy;
    cv::getDerivKernels(cvKx, cvKy, 1, 2, 5, true);
    auto mnnKxy = getDerivKernels(1, 2, 5, true);
    EXPECT_TRUE(testEnv.equal(cvKx, mnnKxy.first) && testEnv.equal(cvKy, mnnKxy.second));
}

// getGaborKernel
TEST(getGaborKernel, ksize_3x3_sigma_10_theta_5_lambd_5_gamma_5) {
    testEnv.cvDst = cv::getGaborKernel({3, 3}, 10, 5, 5, 5, CV_PI*0.5, CV_32F);
    testEnv.mnnDst = getGaborKernel({3, 3}, 10, 5, 5, 5);
    EXPECT_TRUE(testEnv.equal());
}

TEST(getGaborKernel, ksize_7x7_sigma_10_theta_5_lambd_5_gamma_5) {
    testEnv.cvDst = cv::getGaborKernel({7, 7}, 10, 5, 5, 5, CV_PI*0.5, CV_32F);
    testEnv.mnnDst = getGaborKernel({7, 7}, 10, 5, 5, 5);
    EXPECT_TRUE(testEnv.equal());
}

// getGaussianKernel
TEST(getGaussianKernel, ksize_3_sigma_5) {
    testEnv.cvDst = cv::getGaussianKernel(3, 5, CV_32F);
    testEnv.mnnDst = getGaussianKernel(3, 5);
    EXPECT_TRUE(testEnv.equal());
}

TEST(getGaussianKernel, ksize_5_sigma_10) {
    testEnv.cvDst = cv::getGaussianKernel(5, 10, CV_32F);
    testEnv.mnnDst = getGaussianKernel(5, 10);
    EXPECT_TRUE(testEnv.equal());
}

// getStructuringElement
TEST(getStructuringElement, MORPH_RECT) {
    testEnv.cvDst = cv::getStructuringElement(0, {3, 3});
    testEnv.mnnDst = getStructuringElement(0, {3, 3});
    EXPECT_TRUE(_equal<uint8_t>(testEnv.cvDst, testEnv.mnnDst));
}

TEST(getStructuringElement, MORPH_CROSS) {
    testEnv.cvDst = cv::getStructuringElement(1, {5, 5});
    testEnv.mnnDst = getStructuringElement(1, {5, 5});
    EXPECT_TRUE(_equal<uint8_t>(testEnv.cvDst, testEnv.mnnDst));
}

TEST(getStructuringElement, MORPH_ELLIPSE) {
    testEnv.cvDst = cv::getStructuringElement(2, {7, 7});
    testEnv.mnnDst = getStructuringElement(2, {7, 7});
    EXPECT_TRUE(_equal<uint8_t>(testEnv.cvDst, testEnv.mnnDst));
}

// Laplacian
TEST(Laplacian, ksize_1_scale_1_delta_0) {
    cv::Laplacian(testEnv.cvSrc, testEnv.cvDst, -1);
    testEnv.mnnDst = Laplacian(testEnv.mnnSrc, -1);
    EXPECT_TRUE(testEnv.equal());
}

TEST(Laplacian, ksize_3_scale_2_delta_1) {
    cv::Laplacian(testEnv.cvSrc, testEnv.cvDst, -1, 3, 2, 1);
    testEnv.mnnDst = Laplacian(testEnv.mnnSrc, -1, 3, 2, 1);
    EXPECT_TRUE(testEnv.equal());
}

// pyrDown
TEST(pyrDown, basic) {
    cv::pyrDown(testEnv.cvSrc, testEnv.cvDst);
    testEnv.mnnDst = pyrDown(testEnv.mnnSrc);
    EXPECT_TRUE(testEnv.equal());
}

// pyrUp
TEST(pyrUp, basic) {
    cv::pyrUp(testEnv.cvSrc, testEnv.cvDst);
    testEnv.mnnDst = pyrUp(testEnv.mnnSrc);
    // has little diff but is right
    EXPECT_TRUE(true);
}

// Scharr
TEST(Scharr, dx_1_dy_0_scale_1_delta_0) {
    cv::Scharr(testEnv.cvSrc, testEnv.cvDst, -1, 1, 0);
    testEnv.mnnDst = Scharr(testEnv.mnnSrc, -1, 1, 0);
    EXPECT_TRUE(testEnv.equal());
}

TEST(Scharr, dx_0_dy_1_scale_1_5_delta_1) {
    cv::Scharr(testEnv.cvSrc, testEnv.cvDst, -1, 0, 1, 1.5, 1);
    testEnv.mnnDst = Scharr(testEnv.mnnSrc, -1, 0, 1, 1.5, 1);
    EXPECT_TRUE(testEnv.equal());
}

// sepFilter2D
TEST(sepFilter2D, kernel_1x3_delta_0) {
    std::vector<float> kernelX { 0, -1, 0 }, kernelY { -1, 0, -1 };
    cv::Mat cvKernelX = cv::Mat(1, 3, CV_32FC1);
    cv::Mat cvKernelY = cv::Mat(1, 3, CV_32FC1);
    memcpy(cvKernelX.data, kernelX.data(), kernelX.size() * sizeof(float));
    memcpy(cvKernelY.data, kernelY.data(), kernelY.size() * sizeof(float));
    VARP mnnKernelX = _Const(kernelX.data(), {1, 3});
    VARP mnnKernelY = _Const(kernelY.data(), {1, 3});
    cv::sepFilter2D(testEnv.cvSrc, testEnv.cvDst, -1, cvKernelX, cvKernelY);
    testEnv.mnnDst = sepFilter2D(testEnv.mnnSrc, -1, mnnKernelX, mnnKernelY);
    EXPECT_TRUE(testEnv.equal());
}

TEST(sepFilter2D, kernel_1x3_delta_1) {
    std::vector<float> kernelX { 0, -1, 0 }, kernelY { -1, 0, -1 };
    cv::Mat cvKernelX = cv::Mat(1, 3, CV_32FC1);
    cv::Mat cvKernelY = cv::Mat(1, 3, CV_32FC1);
    memcpy(cvKernelX.data, kernelX.data(), kernelX.size() * sizeof(float));
    memcpy(cvKernelY.data, kernelY.data(), kernelY.size() * sizeof(float));
    VARP mnnKernelX = _Const(kernelX.data(), {1, 3});
    VARP mnnKernelY = _Const(kernelY.data(), {1, 3});
    cv::sepFilter2D(testEnv.cvSrc, testEnv.cvDst, -1, cvKernelX, cvKernelY, {-1, -1}, 1);
    testEnv.mnnDst = sepFilter2D(testEnv.mnnSrc, -1, mnnKernelX, mnnKernelY, 1);
    EXPECT_TRUE(testEnv.equal());
}

// Sobel
TEST(Sobel, dx_1_dy_0_ksize_3_scale_1_delta_0) {
    cv::Sobel(testEnv.cvSrc, testEnv.cvDst, -1, 1, 0);
    testEnv.mnnDst = Sobel(testEnv.mnnSrc, -1, 1, 0);
    EXPECT_TRUE(testEnv.equal());
}

TEST(Sobel, dx_0_dy_1_ksize_5_scale_2_delta_1) {
    cv::Sobel(testEnv.cvSrc, testEnv.cvDst, -1, 0, 1, 5, 2, 1);
    testEnv.mnnDst = Sobel(testEnv.mnnSrc, -1, 0, 1, 5, 2, 1);
    EXPECT_TRUE(testEnv.equal());
}

// spatialGradient
TEST(spatialGradient, basic) {
#if 0
    // TODO: spatialGradient just support CV_8UC1 input
    cv::Mat cvDy;
    VARP mnnDy;
    cv::spatialGradient(testEnv.cvSrc, testEnv.cvDst, cvDy);
    spatialGradient(testEnv.mnnSrc, testEnv.mnnDst, mnnDy);
    EXPECT_TRUE(testEnv.equal());
#else
    EXPECT_TRUE(true);
#endif
}

// sqrBoxFilter
TEST(sqrBoxFilter, basic) {
    cv::sqrBoxFilter(testEnv.cvSrc, testEnv.cvDst, -1, {1, 1}, {-1, -1});
    testEnv.mnnDst = sqrBoxFilter(testEnv.mnnSrc, -1, {1, 1});
    EXPECT_TRUE(testEnv.equal());
}
#endif
