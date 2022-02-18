//
//  ImageProcessFunction.hpp
//  MNN
//
//  Created by MNN on 2021/10/29.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef ImageProcessFunction_hpp
#define ImageProcessFunction_hpp
#include <MNN/ImageProcess.hpp>
#include <stdio.h>
#include <memory>

// blitter functions
void MNNGRAYToC4(const unsigned char* source, unsigned char* dest, size_t count);
void MNNGRAYToC3(const unsigned char* source, unsigned char* dest, size_t count);
void MNNC3ToC4(const unsigned char* source, unsigned char* dest, size_t count);
void MNNRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void MNNRGBAToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void MNNRGBToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void MNNBGRAToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void MNNBGRAToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void MNNRGBAToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void MNNC3ToYUV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool yuv);
void MNNC3ToXYZ(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void MNNC3ToHSV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool full);
void MNNC3ToBGR555(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void MNNC3ToBGR565(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void MNNRGBToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void MNNBRGToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void MNNNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count);
// float blitter functions
void MNNC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void MNNC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void MNNC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void MNNC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void MNNC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
// simple blitter functions
inline void MNNCopyC1(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, count * sizeof(unsigned char));
}
inline void MNNCopyC4(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 4 * count * sizeof(unsigned char));
}
inline void MNNCopyC3(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 3 * count * sizeof(unsigned char));
}
inline void MNNRGBToCrCb(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToYUV(source, dest, count, false, false);
}
inline void MNNBGRToCrCb(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToYUV(source, dest, count, true, false);
}
inline void MNNRGBToYUV(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToYUV(source, dest, count, false, true);
}
inline void MNNBGRToYUV(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToYUV(source, dest, count, true, true);
}
inline void MNNRGBToXYZ(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToXYZ(source, dest, count, false);
}
inline void MNNBGRToXYZ(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToXYZ(source, dest, count, true);
}
static void MNNRGBToHSV(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToHSV(source, dest, count, false, false);
}
inline void MNNBGRToHSV(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToHSV(source, dest, count, true, false);
}
static void MNNRGBToHSV_FULL(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToHSV(source, dest, count, false, true);
}
inline void MNNBGRToHSV_FULL(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToHSV(source, dest, count, true, true);
}
inline void MNNRGBToBGR555(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToBGR555(source, dest, count, false);
}
inline void MNNBGRToBGR555(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToBGR555(source, dest, count, true);
}
inline void MNNRGBToBGR565(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToBGR565(source, dest, count, false);
}
inline void MNNBGRToBGR565(const unsigned char* source, unsigned char* dest, size_t count) {
    MNNC3ToBGR565(source, dest, count, true);
}
// sampler
void MNNSamplerC4Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerC3Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerC1Bilinear(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerNearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp);
void MNNSamplerC4Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerC1Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerC3Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerCopyCommon(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                          size_t count, size_t iw, size_t ih, size_t yStride, int bpp);
inline void MNNSamplerC1Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 1);
}
inline void MNNSamplerC3Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 3);
}
inline void MNNSamplerC4Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    MNNSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 4);
}
void MNNSamplerI420Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerI420Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerNV21Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerNV21Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerNV12Copy(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void MNNSamplerNV12Nearest(const unsigned char* source, unsigned char* dest, MNN::CV::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
// draw blit
void MNNC1blitH(const unsigned char* source, unsigned char* dest, size_t count);
void MNNC3blitH(const unsigned char* source, unsigned char* dest, size_t count);
void MNNC4blitH(const unsigned char* source, unsigned char* dest, size_t count);
#endif /* ImageProcessFunction_hpp */
