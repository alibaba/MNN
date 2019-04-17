//
//  CPUTensorConvert.hpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTensorConvert_hpp
#define CPUTensorConvert_hpp

#include "Execution.hpp"
#include "Tensor_generated.h"

namespace MNN {

class CPUTensorConverter : public Execution {
public:
    CPUTensorConverter(Backend* b, MNN_DATA_FORMAT source, MNN_DATA_FORMAT dest) : Execution(b) {
        mSource = source;
        mDest   = dest;
    }
    virtual ~CPUTensorConverter() = default;

    static void NHWC2NC4HW4(const float* source, float* dest, int b, int h, int w, int c);
    static void NC4HW42NHWC(const float* dest, float* source, int b, int h, int w, int c);
    static void NHWC2NCHW(const float* dest, float* source, int b, int h, int w, int c);
    static void NCHW2NHWC(const float* source, float* dest, int b, int h, int w, int c);

    static ErrorCode convert(const Tensor* input, const Tensor* output);
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    MNN_DATA_FORMAT mSource;
    MNN_DATA_FORMAT mDest;
};

} // namespace MNN

#endif /* CPUTensorConvert_hpp */
