//
//  ImageProcessUtils.hpp
//  MNN
//
//  Created by MNN on 2023/09/13.
//  Copyright © 2018, Alibaba Group Holding Limited

#ifndef ImageProcessUtils_hpp
#define ImageProcessUtils_hpp

#include <MNN/ErrorCode.hpp>
#include <MNN/Matrix.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include "backend/cpu/compute/CommonOptFunction.h"


namespace MNN {
typedef void (*BLITTER)(const unsigned char* source, unsigned char* dest, size_t count);
typedef void (*BLIT_FLOAT)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
typedef void (*SAMPLER)(const unsigned char* source, unsigned char* dest, CV::Point* points, size_t sta, size_t count,
                        size_t capacity, size_t iw, size_t ih, size_t yStride);
using namespace MNN::CV;

class ImageProcessUtils {
public:
    struct InsideProperty;
public:
    ImageProcessUtils(const CV::ImageProcess::Config& config, CoreFunctions* core = nullptr);
    ~ImageProcessUtils();
    static void destroy(ImageProcessUtils* imageProcessUtils);

    inline const CV::Matrix& matrix() const {
        return mTransform;
    }
    void setMatrix(const CV::Matrix& matrix);
    void setPadding(uint8_t value) {
        mPaddingValue = value;
    }

    CV::Matrix mTransform;
    CV::Matrix mTransformInvert;
    InsideProperty* mInside;
    uint8_t mPaddingValue = 0;
    
    BLITTER choose(ImageFormat source, ImageFormat dest);
    BLITTER choose(int channelByteSize);
    BLIT_FLOAT choose(ImageFormat format, int dstBpp = 0);
    SAMPLER choose(ImageFormat format, Filter type, bool identity);
    
    void setDraw();
    void draw(uint8_t* img, int w, int h, int c, const int* regions, int num, uint8_t* color);

    ErrorCode transformImage(const uint8_t* source, uint8_t* dst, uint8_t* samplerDest, uint8_t* blitDest, int tileCount, int destBytes, const int32_t* regions);

    ErrorCode selectImageProcer(bool identity=true, bool hasBackend=false, bool isDraw = false);
    ErrorCode execFunc(const uint8_t* source, int stride, void* dest);
    ErrorCode resizeFunc(int ic = 3, int iw = 0, int ih = 0, int oc = 3, int ow = 0, int oh = 0, halide_type_t type = halide_type_of<uint8_t>(), int stride = 0);
    
    
private:
    const CoreFunctions* coreFunctions = nullptr;
};
} // namespace MNN

#endif /* ImageProcessUtils_hpp */
