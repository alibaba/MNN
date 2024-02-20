//
//  NPUBinary.hpp
//  MNN
//
//  Created by MNN on b'2020/10/15'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NPUDEMO_NPUBinary_HPP
#define NPUDEMO_NPUBinary_HPP

#include "NPUCommonExecution.hpp"
#include "NPUBackend.hpp"

namespace MNN {

class NPUBinary : public NPUCommonExecution {
public:
    template<class T>
    void BinaryCastIR(string opName, hiai::Operator& input0, hiai::Operator& input1,
        const std::vector<Tensor*>& outputs, int activationType, shared_ptr<T> binary);
    template<class T>
    void BinaryIR(string opName, hiai::Operator& input0, hiai::Operator& input1,
        const std::vector<Tensor*>& outputs, int activationType, shared_ptr<T> binary);
    void OpInsert(int binary_type, string opName, 
                  hiai::Operator& input0, hiai::Operator& input1,
                  const std::vector<Tensor *> &outputs, int activationType);
    NPUBinary(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~NPUBinary() = default;
   
private:
    hiai::op::Const mConst;
    bool flag0 = false;
    bool flag1 = false;
    int32_t inputIndex0 = -1;
    int32_t inputIndex1 = -1;
};
} // namespace MNN

#endif // NPUDEMO_NPUBinary_HPP
