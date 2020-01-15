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

template <typename T>
class CPUBinary : public Execution {
public:
    CPUBinary(Backend *b, int32_t type);
    virtual ~CPUBinary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int32_t mType;
    void (*mElementProc)(float* C, const float* A, const float* B, size_t width, size_t cStride, size_t aStride, size_t bStride, size_t height) = nullptr;
    bool mSupportScale = false;
};
} // namespace MNN
#endif /* CPUBinary_hpp */
