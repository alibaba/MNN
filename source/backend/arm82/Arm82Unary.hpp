//
//  Arm82Unary.hpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Unary_hpp
#define Arm82Unary_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class Arm82Unary : public Execution {
public:
    Arm82Unary(Backend *b, UnaryOpOperation type);
    virtual ~Arm82Unary() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    template <typename Helper> ErrorCode onExecuteInternal(Tensor*, Tensor*);

protected:
    UnaryOpOperation mType;
};
} // namespace MNN
#endif /* Arm82Unary_hpp */
#endif
