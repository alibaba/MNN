//
//  test_env.cpp
//  MNN
//
//  Created by MNN on 2021/08/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TEST_ENV_HPP
#define TEST_ENV_HPP

// macro flags for module test
#define MNN_CODECS_TEST
#define MNN_TEST_COLOR
#define MNN_DRAW_TEST
#define MNN_TEST_FILTER
#define MNN_GEOMETRIC_TEST
#define MNN_MISCELLANEOUS_TEST
#define MNN_STRUCTRAL_TEST
#define MNN_DRAW_TEST
#define MNN_HISTOGRAMS_TEST
#define MNN_CALIB3D_TEST
#define MNN_CORE_TEST

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv/imgproc/imgproc.hpp"

using namespace MNN;
using namespace Express;
using namespace CV;

static const char* img_name = "./imgs/cat.jpg";

template <typename T> static void _dump(std::string name, const T* ptr, std::vector<int> ids, int stride_w, int stride_h) {
    std::cout << name << std::endl;
    for (int i = ids[0]; i < ids[1]; i++) {
        for (int j = ids[2]; j < ids[3]; j++) {
            for (int k = ids[4]; k < ids[5]; k++) {
                if (sizeof(T) != 1) {
                    std::cout << reinterpret_cast<const T*>(ptr)[i * stride_h + j * stride_w + k] << ", ";
                } else {
                    printf("%d, ", static_cast<int>(ptr[i * stride_h + j * stride_w + k]));
                }
            }
        }
        printf("\n");
    }
}

template <typename Tx, typename Ty> static bool _compare(Tx x, Ty y) {
    if (sizeof(Tx) == 1 && sizeof(Ty) == 1) {
        Tx _y = static_cast<Tx>(y);
        return (abs(x - _y) > 2 && static_cast<float>(abs(x - _y)) / std::max(x, _y) > 2e-2);
    }
    Ty _x = static_cast<Ty>(x);
    return abs(_x - y) > 1e-3;
}

template <typename Tx, typename Ty> static bool _equal(cv::Mat cv, VARP mnn) {
    auto xPtr = reinterpret_cast<const Tx*>(cv.data);
    auto yPtr = mnn->readMap<Ty>();
    // _dump<Tx>("cv:", xPtr, {0, 4, 0, 4, 0, 3}, cv.channels(), cv.cols * cv.channels());
    // _dump<Ty>("mnn:", yPtr, {0, 4, 0, 4, 0, 3}, cv.channels(), cv.cols * cv.channels());
    float deta = 0.f;
    auto dims = mnn->getInfo()->dim;
    int size = 1;
    for (int i = 0; i < dims.size(); ++i) {
        size *= dims[i];
    }
    for (int i = 0; i < size; i++) {
        Tx x = xPtr[i];
        Ty y = yPtr[i];
        if (_compare<Tx, Ty>(x, y)) {
            std::cout << i << ": " << +x << ", " << +y << std::endl;
            return false;
        }
    }
    return true;
}
template <typename T> static bool _equal(cv::Mat cv, VARP mnn) { return _equal<T, T>(cv, mnn); }

template <typename T>
class Env {
public:
    Env(std::string name, bool fp) : isFp(fp) {
        auto img = cv::imread(name);
        if (fp) {
            img.convertTo(cvSrc, CV_32FC3);
            cv2mnn(cvSrc, mnnSrc);
        } else {
            cvSrc = img;
            cv2mnn(cvSrc, mnnSrc);
            cv::cvtColor(cvSrc, cvSrcA, cv::COLOR_RGB2RGBA);
            cv2mnn(cvSrcA, mnnSrcA);
            cv::cvtColor(cvSrc, cvSrcG, cv::COLOR_RGB2GRAY);
            cv2mnn(cvSrcG, mnnSrcG);
            cv::cvtColor(cvSrc, cvSrcY, cv::COLOR_RGB2YUV);
            cv2mnn(cvSrcY, mnnSrcY);
        }
    }
    ~Env() = default;
    bool equal() {
        return equal(cvDst, mnnDst);
    }
    
    bool equal(cv::Mat cv, VARP mnn) {
        return _equal<T>(cv, mnn);
    }
    bool equal(cv::Mat cv, Matrix mnn) {
        for (int i = 0; i < cv.elemSize(); i++) {
            float x = reinterpret_cast<const float*>(cv.data)[i], y = mnn.get(i);
            // printf("%f, ", x);
            if (abs(x - y) > 1e-3) {
                std::cout << i << ": " << x << ", " << y << std::endl;
                return false;
            }
        }
        return true;
    }
    /*
    bool equal(cv::Mat cv, MNN::VARP mnn) {
        auto xPtr = reinterpret_cast<const T*>(cv.data);
        auto yPtr = mnn->readMap<T>();
        // dump("cv:", xPtr, {0, 3, 0, 3, 0, 3}, cv.channels(), cv.cols * cv.channels());
        // dump("mnn:", yPtr, {0, 3, 0, 3, 0, 3}, cv.channels(), cv.cols * cv.channels());
        T deta = 0.f;
        for (int i = 0; i < mnn->getInfo()->size; i++) {
            T x = xPtr[i];
            T y = yPtr[i];
            if (isFp) {
                if (abs(x - y) > 1e-3) {
                    std::cout << i << ": " << x << ", " << y << std::endl;
                    return false;
                }
            } else {
                if (abs(x - y) > 2 && static_cast<float>(abs(x - y)) /std::max(x, y) > 2e-2) {
                    std::cout << i << ": " << (int)x << ", " << (int)y << std::endl;
                    return false;
                }
            }
        }
        return true;
    }*/
    void cv2mnn(const cv::Mat& src, VARP& dst) {
        dst = _Input({ src.rows, src.cols, src.channels() }, NHWC, halide_type_of<T>());
        auto inputPtr = dst->writeMap<T>();
        memcpy(inputPtr, src.ptr(0), dst->getInfo()->size * sizeof(T));
        // _dump<T>("src:", inputPtr, {16, 19, 188, 191, 0, 3}, src.channels(), src.cols * src.channels());
    }
public:
    // RGB/BGR, dst, RGBA, GRAY, YUV
    cv::Mat cvSrc, cvDst, cvSrcA, cvSrcG, cvSrcY;
    VARP mnnSrc, mnnDst, mnnSrcA, mnnSrcG, mnnSrcY;
    bool isFp = false;
};

#endif // TEST_ENV_HPP
