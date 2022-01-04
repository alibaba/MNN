//
//  filter.cpp
//  MNN
//
//  Created by MNN on 2021/08/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cv/imgproc/filter.hpp"
#include <cmath>

namespace MNN {
namespace CV {

// Helper Function //////////////////////////////////////////////////////////////
static VARP PadForConv(VARP& src, int kh, int kw, int padMode) {
    int padh = (kh - 1) / 2;
    int padw = (kw - 1) / 2;
    std::vector<int> padVals { 0, 0, padh, padh, padw, padw, 0, 0};
    return _Pad(src, _Const(padVals.data(), {static_cast<int>(padVals.size())}), static_cast<Express::PadValueMode>(padMode));
}

template <typename T>
static std::vector<T> VARP2Vec(const VARP& var, int ddepth) {
    const int size = var->getInfo()->size;
    const auto ptr = var->readMap<T>();
    std::vector<float> vec(size * ddepth);
    for (int i = 0; i < ddepth; i++) {
        ::memcpy(vec.data() + i * size, ptr, size * sizeof(T));
    }
    return vec;
}

static std::pair<VARP, VARP> getScharrKernels(int dx, int dy, bool normalize) {
    MNN_ASSERT( dx >= 0 && dy >= 0 && dx + dy == 1 );
    float K[2][3] = {{ 3, 10, 3 }, { -1, 0, 1 }};
    VARP kx = _Const(&K[dx], {1, 3});
    VARP ky = _Const(&K[dy], {1, 3});
    if (normalize && dx) {
        kx = kx * _Scalar<float>(1 / 32.f);
    }
    if (normalize && dy) {
        ky = ky * _Scalar<float>(1 / 32.f);
    }
    return { kx, ky };
}

static std::pair<VARP, VARP> getSobelKernels(int dx, int dy, int ksize, bool normalize) {
    if(ksize % 2 == 0 || ksize > 31) {
        MNN_ERROR("The kernel size must be odd and not larger than 31" );
    }
    MNN_ASSERT(dx >= 0 && dy >= 0 && dx+dy > 0);
    VARP kx, ky;
    if (ksize == 1 || ksize == 3) {
        ksize = 3;
        float K[3][3] = {{ 1, 2, 1 }, { -1, 0, 1 }, {1, -2, 1}};
        kx = _Const(&K[dx > 2 ? 2 : dx], {1, 3});
        ky = _Const(&K[dy > 2 ? 2 : dy], {1, 3});
    } else {
        auto getKernel = [&ksize](int order) {
            std::vector<float> kernel(ksize + 1);
            // init
            kernel[0] = 1;
            for(int i = 0; i < ksize; i++) {
                kernel[i + 1] = 0;
            }
            // compute
            float oldVal, newVal;
            for(int i = 0; i < ksize - order - 1; i++) {
                oldVal = kernel[0];
                for(int j = 1; j <= ksize; j++) {
                    newVal = kernel[j] + kernel[j - 1];
                    kernel[j - 1] = oldVal;
                    oldVal = newVal;
                }
            }
            for(int i = 0; i < order; i++) {
                oldVal = -kernel[0];
                for(int j = 1; j <= ksize; j++) {
                    newVal = kernel[j - 1] - kernel[j];
                    kernel[j - 1] = oldVal;
                    oldVal = newVal;
                }
            }
            return kernel;
        };
        kx = _Const(getKernel(dx).data(), {1, ksize});
        ky = _Const(getKernel(dy).data(), {1, ksize});
    }
    if (normalize) {
        kx = kx * _Scalar<float>(1./(1 << (ksize - dx -1)));
        ky = ky * _Scalar<float>(1./(1 << (ksize - dy -1)));
    }
    return { kx, ky };
}

static VARP pyr(VARP src, int borderType) {
    static const std::vector<float> kVec {
        0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625,
        0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ,
        0.0234375 , 0.09375   , 0.140625  , 0.09375   , 0.0234375 ,
        0.015625  , 0.0625    , 0.09375   , 0.0625    , 0.015625  ,
        0.00390625, 0.015625  , 0.0234375 , 0.015625  , 0.00390625 };
    return _Unsqueeze(filter2D(src, -1, _Const(kVec.data(), {5, 5}), 0, borderType), {0});
}

// Helper Function //////////////////////////////////////////////////////////////

MNN_PUBLIC void bilateralFilter(VARP src, VARP& dst, int d, double sigmaColor,
                                double sigmaSpace, int borderType) {
    // TODO
    MNN_ERROR("TODO: bilateralFilter");
}

VARP blur(VARP src, Size ksize, int borderType) {
    return boxFilter(src, -1, ksize, true, borderType);
}

VARP boxFilter(VARP src, int ddepth, Size ksize, bool normalize, int borderType) {
    std::vector<float> filter(ksize.area(), normalize ? 1.f / ksize.area() : 1.f);
    auto kernel = _Const(filter.data(), {ksize.height, ksize.width});
    return filter2D(src, ddepth, kernel, 0.f, borderType);
}

VARP dilate(VARP src, VARP kernel, int iterations, int borderType) {
    auto dim = src->getInfo()->dim;
    int height, width, channel;
    if (dim.size() == 3) {
        height = dim[0];
        width = dim[1];
        channel = dim[2];
        src = _Unsqueeze(src, {0});
    } else {
        getVARPSize(src, &height, &width, &channel);
    }
    int kheight, kwidth, kchannel;
    getVARPSize(kernel, &kheight, &kwidth, &kchannel);
    auto padSrc = PadForConv(src, kheight, kwidth, borderType);
    return _Squeeze(_MaxPool(padSrc, {3, 3}), {0});
}

VARP filter2D(VARP src, int ddepth, VARP kernel, double delta, int borderType) {
    auto dim = src->getInfo()->dim;
    int height, width, channel;
    if (dim.size() == 3) {
        height = dim[0];
        width = dim[1];
        channel = dim[2];
        src = _Unsqueeze(src, {0});
    } else {
        getVARPSize(src, &height, &width, &channel);
    }
    const auto ksize = kernel->getInfo()->dim;
    int kheight, kwidth, kchannel;
    getVARPSize(kernel, &kheight, &kwidth, &kchannel);
    auto padSrc = PadForConv(src, kheight, kwidth, borderType);
    ddepth = ddepth < 0 ? channel : ddepth;
    std::vector<float> bias(ddepth, delta);
    return _Squeeze(_Conv(std::move(VARP2Vec<float>(kernel, ddepth)), std::move(bias), padSrc, {channel, ddepth},
                          {kwidth, kheight}, VALID, {1, 1}, {1, 1}, channel), {0});
}

std::pair<VARP, VARP> getDerivKernels(int dx, int dy, int ksize, bool normalize) {
    if (ksize <= 0) {
        return getScharrKernels(dx, dy, normalize);
    } else {
        return getSobelKernels(dx, dy, ksize, normalize);
    }
}


VARP getGaborKernel(Size ksize, double sigma, double theta, double lambd, double gamma, double psi) {
    double sigma_x = sigma;
    double sigma_y = sigma / gamma;
    double c = cos(theta), s = sin(theta);
    int nstds = 3;
    int xmax = ksize.width > 0 ? ksize.width / 2 :
               std::roundf(std::max(std::fabs(nstds * sigma_x * c), std::fabs(nstds * sigma_y * s)));
    int ymax = ksize.height > 0 ? ksize.height / 2 :
               std::roundf(std::max(std::fabs(nstds * sigma_x * s), std::fabs(nstds * sigma_y * c)));
    int xmin = -xmax;
    int ymin = -ymax;
    
    double scale = 1;
    double ex = -0.5 / (sigma_x * sigma_x);
    double ey = -0.5 / (sigma_y * sigma_y);
    double cscale = MNN_PI * 2 / lambd;
    
    int height = ymax - ymin + 1, width = xmax - xmin + 1;
    std::vector<float> vec(height * width);
    for(int y = ymin; y <= ymax; y++) {
        for(int x = xmin; x <= xmax; x++) {
            double xr = x * c + y * s;
            double yr = -x * s + y * c;
            double v = scale * std::exp(ex * xr * xr + ey * yr * yr) * cos(cscale * xr + psi);
            vec[(ymax - y) * width + (xmax - x)] = static_cast<float>(v);
        }
    }
    return _Const(vec.data(), {height, width});
}

VARP getGaussianKernel(int n, double sigma) {
    constexpr int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };
    
    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
                                small_gaussian_tab[n>>1] : 0;

    std::vector<float> kernel(n);
    double sigmaX = sigma > 0 ? sigma : 0.15 * n + 0.35;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;

    for(int i = 0; i < n; i++ ){
        double x = i - (n - 1) * 0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X * x * x);
        kernel[i] = static_cast<float>(t);
        sum += kernel[i];
    }

    sum = 1./sum;
    for(int i = 0; i < n; i++ ) {
        kernel[i] = (float)(kernel[i] * sum);
    }

    return _Const(kernel.data(), {1, n});
}


VARP getStructuringElement(int shape, Size ksize) {
    // shape: MORPH_RECT = 0, MORPH_CROSS = 1, MORPH_ELLIPSE = 2
    std::vector<uint8_t> elem(ksize.area());
    int anchor_x = ksize.width / 2, anchor_y = ksize.height / 2, r, c;
    double inv_r2 = 0;
    if(shape == 2) {
        r = anchor_y;
        c = anchor_x;
        inv_r2 = r ? 1. / ((double)r * r) : 0;
    }
    for(int i = 0; i < ksize.height; i++ ) {
        uint8_t* ptr = elem.data() + i * ksize.width;
        int start_x = 0, end_x = 0;
        if(shape == 0 || (shape == 1 && i == anchor_y)) {
            end_x = ksize.width;
        } else if(shape == 1) {
            start_x = anchor_x, end_x = start_x + 1;
        } else {
            int dy = i - r;
            if (std::abs(dy) <= r) {
                int dx = static_cast<int>(c * std::sqrt((r*r - dy*dy)*inv_r2));
                start_x = std::max(c - dx, 0);
                end_x = std::min(c + dx + 1, ksize.width);
            }
        }
        for(int j = 0; j < ksize.width; j++) {
            ptr[j] = (j >= start_x && j < end_x);
        }
    }
    return _Const(elem.data(), { ksize.height, ksize.width }, NHWC, halide_type_of<uint8_t>());
}

VARP GaussianBlur(VARP src, Size ksize, double sigmaX, double sigmaY, int borderType) {
    auto kernelX = getGaussianKernel(ksize.width, sigmaX);
    VARP kernelY;
    if (!sigmaY || (ksize.height == ksize.width && std::abs(sigmaY - sigmaX) < 0.1)) {
        kernelY = kernelX;
    } else {
        kernelY = getGaussianKernel(ksize.height, sigmaY);
    }
    return sepFilter2D(src, -1, kernelX, kernelY, 0, borderType);
}

VARP Laplacian(VARP src, int ddepth, int ksize, double scale, double delta, int borderType) {
    if (ksize == 1 || ksize == 3) {
        float K[2][9] = {{ 0, 1, 0, 1, -4, 1, 0, 1, 0 }, { 2, 0, 2, 0, -8, 0, 2, 0, 2 }};
        VARP kernel = _Const(&K[ksize == 3], {3, 3});
        if( scale != 1 ) {
            kernel = kernel * _Scalar<float>(scale);
        };
        return filter2D(src, ddepth, kernel, delta, borderType);
    } else {
        // TODO
        MNN_ERROR("TODO: Laplacian ksize > 3");
        return nullptr;
    }
}

VARP pyrDown(VARP src, Size dstsize, int borderType) {
    return _Convert(_Resize(_Convert(pyr(src, borderType), NC4HW4), 0.5, 0.5), NHWC);
}

VARP pyrUp(VARP src, Size dstsize, int borderType) {
    if (src->getInfo()->dim.size() == 3) {
        src = _Unsqueeze(src, {0});
    }
    return pyr(_Convert(_Resize(_Convert(src, NC4HW4), 2, 2), NHWC), borderType);
}

VARP Scharr(VARP src, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
    return Sobel(src, ddepth, dx, dy, -1, scale, delta, borderType);
}

VARP sepFilter2D(VARP src, int ddepth, VARP& kernelX, VARP& kernelY, double delta, int borderType) {
    auto dims = kernelY->getInfo()->dim;
    kernelY = _Reshape(kernelY, {dims[1], dims[0]});
    VARP tmp = filter2D(src, ddepth, kernelX, 0, borderType);
    return filter2D(tmp, ddepth, kernelY, delta, borderType);
}

VARP Sobel(VARP src, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType) {
    auto kxy = getDerivKernels(dx, dy, ksize, false);
    if (scale != 1.0) {
        if (dx) {
            kxy.first = kxy.first * _Scalar<float>(scale);
        } else {
            kxy.second = kxy.second * _Scalar<float>(scale);
        }
    }
    return sepFilter2D(src, ddepth, kxy.first, kxy.second, delta, borderType);
}

std::pair<VARP, VARP> spatialGradient(VARP src, int ksize, int borderType) {
    auto dx = Sobel(src, 1, 1, 0, 3);
    auto dy = Sobel(src, 1, 0, 1, 3);
    return { dx, dy };
}

VARP sqrBoxFilter(VARP src, int ddepth, Size ksize, bool normalize, int borderType) {
    return boxFilter(src * src, ddepth, ksize, normalize, borderType);
}

} // CV
} // MNN
