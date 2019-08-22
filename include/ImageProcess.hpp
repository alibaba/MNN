//
//  ImageProcess.hpp
//  MNN
//
//  Created by MNN on 2018/09/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageProcess_hpp
#define ImageProcess_hpp

#include "ErrorCode.hpp"
#include "Matrix.h"
#include "Tensor.hpp"

namespace MNN {
namespace CV {
enum ImageFormat {
    RGBA = 0,
    RGB,
    BGR,
    GRAY,
    BGRA,
    YUV_NV21 = 11,
};

enum Filter { NEAREST = 0, BILINEAR = 1, BICUBIC = 2 };

enum Wrap { CLAMP_TO_EDGE = 0, ZERO = 1, REPEAT = 2 };

/**
 * handle image process for tensor.
 * step:
 *  1: Do transform compute and get points
 *  2: Sample line and do format convert
 *  3: Turn RGBA to float tensor, and do sub and normalize
 */
class MNN_PUBLIC ImageProcess {
public:
    struct Inside;
    struct Config {
        /** data filter */
        Filter filterType = NEAREST;
        /** format of source data */
        ImageFormat sourceFormat = RGBA;
        /** format of destination data */
        ImageFormat destFormat = RGBA;

        // Only valid if the dest type is float
        float mean[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
        float normal[4] = {1.0f, 1.0f, 1.0f, 1.0f};

        /** edge wrapper */
        Wrap wrap = CLAMP_TO_EDGE;
    };

public:
    /**
     * @brief create image process with given config for given tensor.
     * @param config    given config.
     * @param dstTensor given tensor.
     * @return image processor.
     */
    static ImageProcess* create(const Config& config, const Tensor* dstTensor = nullptr);

    /**
     * @brief create image process with given config for given tensor.
     * @param means given means
     * @param meanCount given means count
     * @param normals   given normals
     * @param normalCount given normal count
     * @param sourceFormat  format of source data
     * @param destFormat    format of destination data
     * @param dstTensor given tensor.
     * @return image processor.
     */
    static ImageProcess* create(const ImageFormat sourceFormat = RGBA, const ImageFormat destFormat = RGBA,
                                const float* means = nullptr, const int meanCount = 0, const float* normals = nullptr,
                                const int normalCount = 0, const Tensor* dstTensor = nullptr);

    ~ImageProcess();

    /**
     * @brief get affine transform matrix.
     * @return affine transform matrix.
     */
    inline const Matrix& matrix() const {
        return mTransform;
    }
    void setMatrix(const Matrix& matrix);

    /**
     * @brief convert source data to given tensor.
     * @param source    source data.
     * @param iw        source width.
     * @param ih        source height.
     * @param stride    number of elements per row. eg: 100 width RGB contains at least 300 elements.
     * @param dest      given tensor.
     * @return result code.
     */
    ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, Tensor* dest);

    /**
     * @brief convert source data to given tensor.
     * @param source    source data.
     * @param iw        source width.
     * @param ih        source height.
     * @param stride    number of elements per row. eg: 100 width RGB contains at least 300 elements.
     * @param dest      dest data.
     * @param ow      output width.
     * @param oh      output height.
     * @param outputBpp      output bpp, if 0, set as the save and config.destFormat.
     * @param outputStride  output stride, if 0, set as ow * outputBpp.
     * @param type  Only support halide_type_of<uint8_t> and halide_type_of<float>.
     * @return result code.
     */
    ErrorCode convert(const uint8_t* source, int iw, int ih, int stride, void* dest, int ow, int oh, int outputBpp = 0,
                      int outputStride = 0, halide_type_t type = halide_type_of<float>());

    /**
     * @brief create tensor with given data.
     * @param w     image width.
     * @param h     image height.
     * @param bpp   bytes per pixel.
     * @param p     pixel data pointer.
     * @return created tensor.
     */
    template <typename T>
    static Tensor* createImageTensor(int w, int h, int bpp, void* p = nullptr) {
        return createImageTensor(halide_type_of<T>(), w, h, bpp, p);
    }
    static Tensor* createImageTensor(halide_type_t type, int w, int h, int bpp, void* p = nullptr);

private:
    ImageProcess(const Config& config);
    Matrix mTransform;
    Matrix mTransformInvert;
    Inside* mInside;
};
} // namespace CV
} // namespace MNN

#endif /* ImageProcess_hpp */
