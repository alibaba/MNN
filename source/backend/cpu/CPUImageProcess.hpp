//
//  CPUImageProcess.hpp
//  MNN
//
//  Created by MNN on 2021/10/27.
//  Copyright © 2018 Alibaba. All rights reserved.
//

#ifndef CPUImageProcess_hpp
#define CPUImageProcess_hpp

#include "backend/cpu/CPUBackend.hpp"
#include <MNN/ImageProcess.hpp>
#include "compute/CommonOptFunction.h"
#include "cv/ImageProcessUtils.hpp"

namespace MNN {

class CPUImageProcess : public Execution {
public:
    CPUImageProcess(const CV::ImageProcess::Config& config, const CoreFunctions* coreFunctions) : Execution(nullptr), coreFunctions(coreFunctions) {
        mImgConfig.filterType = config.filterType;
        mImgConfig.wrap = config.wrap;
        mImgConfig.sourceFormat = config.sourceFormat;
        mImgConfig.destFormat = config.destFormat;
        for (int i = 0; i < 4; i++) {
            mImgConfig.mean[i] = config.mean[i];
            mImgConfig.normal[i] = config.normal[i];
        }
    }
    void setMatrix(CV::Matrix m) {
        transform = m;
        transform.invert(&transformInvert);
        mImgProc->setMatrix(m);
    }
    void setPadVal(uint8_t val) {
        paddingValue = val;
        mImgProc->setPadding(val);
    }
    void setDraw() {
        mImgProc->setDraw();
    }
    CPUImageProcess(Backend *bn, const ImageProcessParam* process) : Execution(bn) {
        coreFunctions = static_cast<CPUBackend*>(backend())->functions();
        draw = process->draw();
        if (draw) {
            return;
        }
        mImgConfig.filterType = (CV::Filter)process->filterType();
        mImgConfig.wrap = (CV::Wrap)process->wrap();
        mImgConfig.sourceFormat = (CV::ImageFormat)process->sourceFormat();
        mImgConfig.destFormat = (CV::ImageFormat)process->destFormat();
        paddingValue = process->paddingValue();
        for (int i = 0; i < 4; i++) {
            mImgConfig.mean[i] = process->mean()->Get(i);
            mImgConfig.normal[i] = process->normal()->Get(i);
        }
        for (int i = 0; i < process->transform()->size(); i++) {
            transform.set(i, process->transform()->Get(i));
        }
        transform.invert(&transformInvert);
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // ~CPUImageProcess();
    // void destroy(CPUImageProcess* pro);
private:
    std::unique_ptr<ImageProcessUtils> mImgProc;
private:
    CV::ImageProcess::Config mImgConfig;
    uint8_t paddingValue = 0;
    CV::Matrix transform, transformInvert;
    const CoreFunctions* coreFunctions = nullptr;
    bool draw = false;
    int mStride = 0;
};
}; // namespace MNN

#endif /* CPUImageProcess_hpp */
