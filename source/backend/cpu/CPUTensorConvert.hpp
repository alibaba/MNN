//
//  CPUTensorConvert.hpp
//  MNN
//
//  Created by MNN on 2018/08/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUTensorConvert_hpp
#define CPUTensorConvert_hpp

#include "core/Execution.hpp"
#include "Tensor_generated.h"

namespace MNN {

class CPUTensorConverter : public Execution {
public:
    CPUTensorConverter(Backend* b) : Execution(b) {
        // Do nothing
    }
    virtual ~CPUTensorConverter() = default;
    static std::tuple<int, int, int> splitDimensions(const halide_buffer_t& ib, MNN_DATA_FORMAT source);
    static ErrorCode convert(const Tensor* input, const Tensor* output);
    static ErrorCode convert(const void* inputRaw, void* outputRaw, MNN_DATA_FORMAT inputFormat, MNN_DATA_FORMAT outputFormat, int batch, int area, int channel, int bytes);
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
};

} // namespace MNN

#endif /* CPUTensorConvert_hpp */
