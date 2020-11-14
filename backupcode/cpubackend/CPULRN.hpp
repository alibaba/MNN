//
//  CPULRN.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULRN_hpp
#define CPULRN_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPULRN : public Execution {
public:
    CPULRN(Backend *backend, int regionType, int localSize, float alpha, float beta);
    virtual ~CPULRN() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    void executeAcrossChannels(const float* srcData, float* dstData, const int width, const int height, const int channels, const float* powfParam);
    void executeWithInChannels(const float* srcData, float* dstData, const int width, const int height, const int channels, const float* powfParam);
    
private:
    Tensor mStorage;
    Tensor mSquare;
    int mRegionType;
    int mLocalSize;
    float mAlpha;
    float mBeta;
};

} // namespace MNN

#endif /* CPULRN_hpp */
