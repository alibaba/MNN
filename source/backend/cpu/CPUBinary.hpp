//
//  CPUBinary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBinary_hpp
#define CPUBinary_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUBinaryFloat : public Execution {
public:
    CPUBinaryFloat(Backend *b, int32_t type);
    virtual ~CPUBinaryFloat() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int32_t mType;
    void (*mElementProc)(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) = nullptr;
    bool mSupportScale = false;
    int mOutside = 1;
    int mInside = 1;
    int mAxis = 1;
};
class CPUBinaryInt : public Execution {
public:
    CPUBinaryInt(Backend *b, int32_t type);
    virtual ~CPUBinaryInt() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int32_t mType;
};
} // namespace MNN
#endif /* CPUBinary_hpp */
