//
//  CPUImageProcess.hpp
//  MNN
//
//  Created by MNN on 2021/10/27.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUImageProcess_hpp
#define CPUImageProcess_hpp

#include <MNN/ImageProcess.hpp>
#include "backend/cpu/CPUBackend.hpp"
#include "compute/CommonOptFunction.h"

namespace MNN {

typedef void (*BLITTER)(const unsigned char* source, unsigned char* dest, size_t count);
typedef void (*BLIT_FLOAT)(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
typedef void (*SAMPLER)(const unsigned char* source, unsigned char* dest, CV::Point* points, size_t sta, size_t count,
                        size_t capacity, size_t iw, size_t ih, size_t yStride);

class CPUImageProcess : public Execution {
public:
    CPUImageProcess(CV::ImageProcess::Config config, const CoreFunctions* coreFunctions) : Execution(nullptr), coreFunctions(coreFunctions) {
        filterType = (FilterType)config.filterType;
        wrap = (WrapType)config.wrap;
        sourceFormat = (ImageFormatType)config.sourceFormat;
        destFormat = (ImageFormatType)config.destFormat;
        for (int i = 0; i < 4; i++) {
            mean[i] = config.mean[i];
            normal[i] = config.normal[i];
        }
    }
    void setMatrix(CV::Matrix m) {
        transform = m;
        transform.invert(&transformInvert);
    }
    void setPadVal(uint8_t val) {
        paddingValue = val;
    }
    CPUImageProcess(Backend *bn, const ImageProcessParam* process) : Execution(bn) {
        filterType = process->filterType();
        wrap = process->wrap();
        sourceFormat = process->sourceFormat();
        destFormat = process->destFormat();
        paddingValue = process->paddingValue();
        for (int i = 0; i < 4; i++) {
            mean[i] = process->mean()->Get(i);
            normal[i] = process->normal()->Get(i);
        }
        for (int i = 0; i < process->transform()->size(); i++) {
            transform.set(i, process->transform()->Get(i));
        }
        transform.invert(&transformInvert);
        coreFunctions = static_cast<CPUBackend*>(backend())->functions();
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
private:
    BLITTER choose(ImageFormatType source, ImageFormatType dest);
    BLIT_FLOAT choose(ImageFormatType format, int dstBpp = 0);
    SAMPLER choose(ImageFormatType format, FilterType type, bool identity);
private:
    FilterType filterType;
    WrapType wrap;
    ImageFormatType sourceFormat, destFormat;
    float mean[4]   = {0.0f, 0.0f, 0.0f, 0.0f};
    float normal[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint8_t paddingValue = 0;
    int ih, iw, ic, oh, ow, oc;
    halide_type_t dtype;
    CV::Matrix transform, transformInvert;
    SAMPLER sampler;
    BLITTER blitter = nullptr;
    BLIT_FLOAT blitFloat = nullptr;
    std::shared_ptr<Tensor> cacheBuffer, cacheBufferRGBA;
    std::unique_ptr<uint8_t[]> samplerBuffer, blitBuffer;
    uint8_t* samplerDest = nullptr, *blitDest = nullptr;
    const CoreFunctions* coreFunctions = nullptr;
};
}; // namespace MNN

#endif /* CPUImageProcess_hpp */
