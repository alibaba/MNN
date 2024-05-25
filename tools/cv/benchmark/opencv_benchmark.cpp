//
//  opencv_benchmark.cpp
//  MNN
//
//  Created by MNN on 2022/06/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv/imgproc/imgproc.hpp"
#include "cv/calib3d.hpp"
#ifdef MNN_IMGCODECS
#include "cv/imgcodecs.hpp"
#endif

using namespace MNN;
using namespace Express;
using namespace CV;

constexpr int LOOP = 10;

static std::vector<std::string> functions;
static std::vector<double> times;
constexpr const char* path = "./imgs/cat.jpg";

template <typename T>
VARP cv2mnn(const cv::Mat& src) {
    VARP dst = _Input({ src.rows, src.cols, src.channels() }, NHWC, halide_type_of<T>());
    auto inputPtr = dst->writeMap<T>();
    memcpy(inputPtr, src.ptr(0), dst->getInfo()->size * sizeof(T));
    return dst;
}

#define arg_concat_impl(x, y) x ## y
#define arg_concat(x, y) arg_concat_impl(x, y)
#define arg_switch_0(CASE0, CASE1, CASE2, CASE3) CASE0
#define arg_switch_1(CASE0, CASE1, CASE2, CASE3) CASE1
#define arg_switch_2(CASE0, CASE1, CASE2, CASE3) CASE2
#define arg_switch_3(CASE0, CASE1, CASE2, CASE3) CASE3
// just support COND = [0, 1, 2, 3]
#define arg_switch(COND, CASE0, CASE1, CASE2, CASE3) arg_concat(arg_switch_, COND)(CASE0, CASE1, CASE2, CASE3)

#define BENCH_IMPL(mode, func, ...)\
    auto t1 =  std::chrono::high_resolution_clock::now();\
    for (int i = 0; i < LOOP; i++) {\
arg_switch(mode, cv::func(__VA_ARGS__);, auto dst = func(__VA_ARGS__);dst->readMap<void>();, auto dst = func(__VA_ARGS__);dst[0]->readMap<void>();, func(__VA_ARGS__);)\
    }\
    auto t2 =  std::chrono::high_resolution_clock::now();\
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (1000. * LOOP);\
    times.push_back(duration); \

#define BENCHMARK_NAME(mode, name, func, ...) \
do {\
    BENCH_IMPL(mode, func, __VA_ARGS__)\
    functions.emplace_back(#name);\
} while(0);

#define BENCHMARK(mode, func, ...) \
do {\
    BENCH_IMPL(mode, func, __VA_ARGS__)\
    functions.emplace_back(#func); \
} while(0);

#define BENCHMARK_CV(func, ...) BENCHMARK(0, func, __VA_ARGS__)
#define BENCHMARK_MNN(func, ...) BENCHMARK(1, func, __VA_ARGS__)

void color(cv::Mat cvimg, VARP mnnimg) {
    cv::Mat dst;
#define CVTCOLOR(code)\
    BENCHMARK_NAME(0, code, cvtColor, cvimg, dst, cv::COLOR_##code)\
    BENCHMARK_NAME(1, code, cvtColor, mnnimg, COLOR_##code)
    CVTCOLOR(RGB2BGR)
    CVTCOLOR(RGB2GRAY)
    CVTCOLOR(RGB2RGBA)
    CVTCOLOR(RGB2BGRA)
    CVTCOLOR(RGB2YUV)
    CVTCOLOR(RGB2XYZ)
    CVTCOLOR(RGB2HSV)
    CVTCOLOR(RGB2HSV_FULL)
    CVTCOLOR(RGB2BGR555)
    CVTCOLOR(RGB2BGR565)
}

void filter(cv::Mat cvimg, VARP mnnimg) {
    cv::Mat dst;
    // blur
    BENCHMARK_CV(blur, cvimg, dst, {3, 3});
    BENCHMARK_MNN(blur, mnnimg, {3, 3});
    // boxFilter
    BENCHMARK_CV(boxFilter, cvimg, dst, -1, {3, 3});
    BENCHMARK_MNN(boxFilter, mnnimg, -1, {3, 3});
    // dilate
    BENCHMARK_CV(dilate, cvimg, dst, cv::getStructuringElement(0, {3, 3}));
    BENCHMARK_MNN(dilate, mnnimg, getStructuringElement(0, {3, 3}));
    // filter2D
    std::vector<float> kernel { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
    cv::Mat cvKernel = cv::Mat(3, 3, CV_32FC1);
    memcpy(cvKernel.data, kernel.data(), kernel.size() * sizeof(float));
    VARP mnnKernel = _Const(kernel.data(), {3, 3});
    BENCHMARK_CV(filter2D, cvimg, dst, -1, cvKernel);
    BENCHMARK_MNN(filter2D, mnnimg, -1, mnnKernel);
    // boxFilter
    BENCHMARK_CV(GaussianBlur, cvimg, dst, {3, 3}, 10);
    BENCHMARK_MNN(GaussianBlur, mnnimg, {3, 3}, 10);
    // getDerivKernels
    BENCHMARK_CV(getDerivKernels, dst, dst, 1, 2, 1);
    BENCHMARK(3, getDerivKernels, 1, 2, 1);
    // getGaborKernel
    BENCHMARK_CV(getGaborKernel, {3, 3}, 10, 5, 5, 5, CV_PI*0.5, CV_32F);
    BENCHMARK_MNN(getGaborKernel, {3, 3}, 10, 5, 5, 5);
    // getGaussianKernel
    BENCHMARK_CV(getGaussianKernel, 3, 5, CV_32F);
    BENCHMARK_MNN(getGaussianKernel, 3, 5);
    // getStructuringElement
    BENCHMARK_CV(getStructuringElement, 0, {3, 3});
    BENCHMARK_MNN(getStructuringElement, 0, {3, 3});
    // Laplacian
    BENCHMARK_CV(Laplacian, cvimg, dst, -1);
    BENCHMARK_MNN(Laplacian, mnnimg, -1);
    // pyrDown
    BENCHMARK_CV(pyrDown, cvimg, dst);
    BENCHMARK_MNN(pyrDown, mnnimg);
    // pyrUp
    BENCHMARK_CV(pyrUp, cvimg, dst);
    BENCHMARK_MNN(pyrUp, mnnimg);
    // Scharr
    BENCHMARK_CV(Scharr, cvimg, dst, -1, 1, 0);
    BENCHMARK_MNN(Scharr, mnnimg, -1, 1, 0);
    // sepFilter2D
    std::vector<float> kernelX { 0, -1, 0 }, kernelY { -1, 0, -1 };
    cv::Mat cvKernelX = cv::Mat(1, 3, CV_32FC1);
    cv::Mat cvKernelY = cv::Mat(1, 3, CV_32FC1);
    memcpy(cvKernelX.data, kernelX.data(), kernelX.size() * sizeof(float));
    memcpy(cvKernelY.data, kernelY.data(), kernelY.size() * sizeof(float));
    VARP mnnKernelX = _Const(kernelX.data(), {1, 3});
    VARP mnnKernelY = _Const(kernelY.data(), {1, 3});
    BENCHMARK_CV(sepFilter2D, cvimg, dst, -1, cvKernelX, cvKernelY);
    BENCHMARK_MNN(sepFilter2D, mnnimg, -1, mnnKernelX, mnnKernelY);
    // Sobel
    BENCHMARK_CV(Sobel, cvimg, dst, -1, 1, 0);
    BENCHMARK_MNN(Sobel, mnnimg, -1, 1, 0);
    // sqrBoxFilter
    BENCHMARK_CV(sqrBoxFilter, cvimg, dst, -1, {1, 1}, {-1, -1});
    BENCHMARK_MNN(sqrBoxFilter, mnnimg, -1, {1, 1});
}

void geometric(cv::Mat cvimg, VARP mnnimg) {
    cv::Mat dst;
    // getAffineTransform
    float points[] = { 50, 50, 200, 50, 50, 200, 10, 100, 200, 20, 100, 250, 100, 20, 50, 100};
    cv::Point2f cvSrc[4], cvDst[4];
    memcpy(cvSrc, points, 8 * sizeof(float));
    memcpy(cvDst, points + 8, 8 * sizeof(float));
    Point mnnSrc[4], mnnDst[4];
    memcpy(mnnSrc, points, 8 * sizeof(float));
    memcpy(mnnDst, points + 8, 8 * sizeof(float));
    BENCHMARK_CV(getAffineTransform, cvSrc, cvDst);
    BENCHMARK(3, getAffineTransform, mnnSrc, mnnDst);
    // getPerspectiveTransform
    BENCHMARK_CV(getPerspectiveTransform, cvSrc, cvDst);
    BENCHMARK(3, getPerspectiveTransform, mnnSrc, mnnDst);
    // getRotationMatrix2D
    cv::Point2f cvCenter {10, 10};
    Point mnnCenter {10, 10};
    BENCHMARK_CV(getRotationMatrix2D, cvCenter, 50, 0.6);
    BENCHMARK(3, getRotationMatrix2D, mnnCenter, 50, 0.6);
    // getRectSubPix
    BENCHMARK_CV(getRectSubPix, cvimg, {11, 11}, cvCenter, dst);
    BENCHMARK_MNN(getRectSubPix, mnnimg, {11, 11}, mnnCenter);
    // invertAffineTransform
    std::vector<float> M { 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    cv::Mat cvM = cv::Mat(2, 3, CV_32FC1), cvMDst = cv::Mat(2, 3, CV_32FC1);
    memcpy(cvM.data, M.data(), M.size() * sizeof(float));
    Matrix mnnM;
    for (int i = 0; i < M.size(); i++) mnnM.set(i, M[i]);
    BENCHMARK_CV(invertAffineTransform, cvM, cvMDst);
    BENCHMARK(3, invertAffineTransform, mnnM);
    // resize
    BENCHMARK_CV(resize, cvimg, dst, cv::Size(), 2, 2);
    BENCHMARK_MNN(resize, mnnimg, {}, 2, 2);
    // warpAffine
    BENCHMARK_CV(warpAffine, cvimg, dst, cvM, {480, 360});
    BENCHMARK_MNN(warpAffine, mnnimg, mnnM, {480, 360});
}

void miscellaneous(cv::Mat cvimg, VARP mnnimg) {
    cv::Mat dst;
    // blendLinear
    int weightSize = cvimg.rows * cvimg.cols;
    std::vector<float> weight1(weightSize, 0.6), weight2(weightSize, 0.7);
    std::vector<float> mnnweight1(weightSize), mnnweight2(weightSize);
    cv::Mat cvWeight1 = cv::Mat(cvimg.rows, cvimg.cols, CV_32FC1);
    cv::Mat cvWeight2 = cv::Mat(cvimg.rows, cvimg.cols, CV_32FC1);
    memcpy(cvWeight1.data, weight1.data(), weight1.size() * sizeof(float));
    memcpy(cvWeight2.data, weight2.data(), weight2.size() * sizeof(float));
    auto mnnWeight1 = cv2mnn<float>(cvWeight1);
    auto mnnWeight2 = cv2mnn<float>(cvWeight2);

    BENCHMARK_CV(blendLinear, cvimg, cvimg, cvWeight1, cvWeight2, dst);
    BENCHMARK_MNN(blendLinear, mnnimg, mnnimg, mnnWeight1, mnnWeight2);
    // threshold
    BENCHMARK_CV(threshold, cvimg, dst, 50, 20, cv::THRESH_BINARY);
    BENCHMARK_MNN(threshold, mnnimg, 50, 20, cv::THRESH_BINARY);
}

void structral(cv::Mat cvimg, VARP mnnimg) {
    static std::vector<float> img = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,1,0,0,0,0,0,0,0,0,0,
        0,0,1,1,1,1,1,1,1,0,0,0,0,
        0,0,1,0,0,1,0,0,0,1,1,0,0,
        0,0,1,0,0,1,0,0,1,0,0,0,0,
        0,0,1,0,0,1,0,0,1,0,0,0,0,
        0,0,1,1,1,1,1,1,1,0,0,0,0,
        0,0,0,1,0,0,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0
    };
    // findContours
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    VARP x = _Const(img.data(), {1, 11, 13, 1}, NHWC);
    cv::Mat mask = cv::Mat(11, 13, CV_8UC1);
    ::memcpy(mask.data, img.data(), img.size() * sizeof(uchar));
    BENCHMARK_CV(findContours, mask, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    BENCHMARK(2, findContours, x, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // contourArea
    std::vector<cv::Point2i> cv_contour = { {0, 0}, {10, 0}, {10, 10}, {5, 4}};
    VARP mnn_contour = _Const(cv_contour.data(), {4, 2}, NHWC, halide_type_of<int>());
    BENCHMARK_CV(contourArea, cv_contour);
    BENCHMARK(3, contourArea, mnn_contour);
    // convexHull
    std::vector<int> y;
    BENCHMARK_CV(convexHull, cv_contour, y, false, false);
    BENCHMARK(3, convexHull, mnn_contour, false, false);
    // minAreaRect
    BENCHMARK_CV(minAreaRect, cv_contour);
    BENCHMARK(3, minAreaRect, mnn_contour);
    // boundingRect
    BENCHMARK_CV(boundingRect, cv_contour);
    BENCHMARK(3, boundingRect, mnn_contour);
    // connectedComponentsWithStats
    cv::Mat cv_label, cv_statsv, cv_centroids;
    VARP mnn_label, mnn_statsv, mnn_centroids;
    BENCHMARK_CV(connectedComponentsWithStats, mask, cv_label, cv_statsv, cv_centroids);
    BENCHMARK(3, connectedComponentsWithStats, x, mnn_label, mnn_statsv, mnn_centroids);
    // boxPoints
    BENCHMARK_CV(boxPoints, cv::minAreaRect(cv_contour), cv_label);
    BENCHMARK(3, boxPoints, minAreaRect(mnn_contour));
}

void draw(cv::Mat cvimg, VARP mnnimg) {
#define DRAW(func, ...)\
    BENCHMARK_CV(func, cvimg, __VA_ARGS__)\
    BENCHMARK(3, func, mnnimg, __VA_ARGS__)
    // arrowedLine
    DRAW(arrowedLine, {10, 10}, {300, 200}, {0, 0, 255}, 1)
    // circle
    DRAW(circle, {50, 50}, 10, {0, 0, 255}, 1)
    // line
    DRAW(line, {10, 10}, {50, 50}, {0, 0, 255}, 5)
    // rectangle
    DRAW(rectangle, {10, 10}, {200, 300}, {0, 0, 255}, 1)
    // drawContours
    cv::Mat gray, binary;
    cv::cvtColor(cvimg, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> cv_contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, cv_contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<Point>> mnn_contours(cv_contours.size());
    for (int i = 0; i < cv_contours.size(); i++) {
        for (int j = 0; j < cv_contours[i].size(); j++) {
            Point p;
            p.set(cv_contours[i][j].x, cv_contours[i][j].y);
            mnn_contours[i].push_back(p);
        }
    }
    BENCHMARK_CV(drawContours, cvimg, cv_contours, -1, {0, 0, 255}, -1)
    BENCHMARK(3, drawContours, mnnimg, mnn_contours, -1, {0, 0, 255}, -1)
    // fillPoly
    BENCHMARK_CV(fillPoly, cvimg, cv_contours, {0, 0, 255})
    BENCHMARK(3, fillPoly, mnnimg, mnn_contours, {0, 0, 255})
}

void histogram(cv::Mat cvimg, VARP mnnimg) {
    std::vector<cv::Mat> images {cvimg};
    std::vector<int> histSize {256};
    std::vector<int> channels {0};
    std::vector<float> ranges {0., 256.};
    cv::Mat cvDest;
    // solvePnP
    BENCHMARK_CV(calcHist, images, channels, cv::Mat(), cvDest, histSize, ranges)
    BENCHMARK_MNN(calcHist, {mnnimg}, channels, nullptr, histSize, ranges)
}


void codecs(cv::Mat cvimg, VARP mnnimg) {
#ifdef MNN_IMGCODECS
    // imread
    BENCHMARK_CV(imread, path)
    BENCHMARK_MNN(imread, path)
    // imwrite
    BENCHMARK_CV(imwrite, "cv.jpg", cvimg)
    BENCHMARK(3, imwrite, "mnn.jpg", mnnimg)
#endif
}

void calib3d(cv::Mat cvimg, VARP mnnimg) {
    float model_points[18] = {
        0.0, 0.0, 0.0, 0.0, -330.0, -65.0, -225.0, 170.0, -135.0,
        225.0, 170.0, -135.0, -150.0, -150.0, -125.0, 150.0, -150.0, -125.0
    };
    float image_points[12] = {
        359, 391, 399, 561, 337, 297, 513, 301, 345, 465, 453, 469
    };
    float camera_matrix[9] = {
        1200, 0, 600, 0, 1200, 337.5, 0, 0, 1
    };
    float dist_coeffs[4] = { 0, 0, 0, 0 };
    VARP mnnObj = _Const(model_points, {6, 3});
    VARP mnnImg = _Const(image_points, {6, 2});
    VARP mnnCam = _Const(camera_matrix, {3, 3});
    VARP mnnCoe = _Const(dist_coeffs, {4, 1});
    cv::Mat cvObj = cv::Mat(6, 3, CV_32F, model_points);
    cv::Mat cvImg = cv::Mat(6, 2, CV_32F, image_points);
    cv::Mat cvCam = cv::Mat(3, 3, CV_32F, camera_matrix);
    cv::Mat cvCoe = cv::Mat(4, 1, CV_32F, dist_coeffs);
    std::vector<float> rv(3), tv(3);
    cv::Mat rvecs(rv),tvecs(tv);
    // solvePnP
    BENCHMARK_CV(solvePnP, cvObj, cvImg, cvCam, cvCoe, rvecs, tvecs, false, cv::SOLVEPNP_SQPNP)
    BENCHMARK(3, solvePnP, mnnObj, mnnImg, mnnCam, mnnCoe)
}

void printLine() {
    std::cout << "+----------------------------+----------+----------+" << std::endl;
}
void printLine(const char* col0, const char* col1, const char* col2) {
    std::cout << std::setiosflags(std::ios::left) << "|" << std::setw(28) << col0 << "|" << std::setw(10) << col1 << "|" << std::setw(10) << col2 << "|" << std::endl;
}
void printLine(const std::string& func, double t1, double t2) {
    std::cout << std::setiosflags(std::ios::left) << "|" << std::setw(28) << func << "|" << std::setw(10) << t1 << "|" << std::setw(10) << t2 << "|" << std::endl;
}
void log() {
    int count = times.size() / 2;
    printLine();
    printLine("function", "opencv", "MNN.cv");
    printLine();
    double cv_sum = 0., mnn_sum = 0.;
    for (int i = 0; i < count; i++) {
        auto func = functions[i * 2];
        double cv = times[i * 2];
        double mnn = times[i * 2 + 1];
        cv_sum += cv;
        mnn_sum += mnn;
        printLine(func, cv, mnn);
    }
    printLine();
    printLine("avg", cv_sum/count, mnn_sum/count);
    printLine();
}

int main(int argc, char** argv) {
    printf("opencv benchmark\n");
    cv::setNumThreads(1);
    // uint8
    auto img_uchar = cv::imread(path);
    auto mnn_uchar = cv2mnn<uint8_t>(img_uchar);
    // fp32
    cv::Mat img_fp32;
    img_uchar.convertTo(img_fp32, CV_32FC3);
    auto mnn_fp32 = cv2mnn<float>(img_fp32);
    color(img_uchar, mnn_uchar);
    filter(img_fp32, mnn_fp32);
    geometric(img_uchar, mnn_uchar);
    miscellaneous(img_fp32, mnn_fp32);
    structral(img_uchar, mnn_uchar);
    draw(img_uchar, mnn_uchar);
    codecs(img_uchar, mnn_uchar);
    calib3d(img_uchar, mnn_uchar);
    histogram(img_uchar, mnn_uchar);
    log();
    return 0;
}
