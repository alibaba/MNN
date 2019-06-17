//
//  StrassenMatmulComputor.hpp
//  MNN
//
//  Created by MNN on 2019/02/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StrassenMatmulComputor_hpp
#define StrassenMatmulComputor_hpp

#include <functional>
#include "Backend.hpp"
namespace MNN {
class StrassenMatrixComputor {
public:
    StrassenMatrixComputor(Backend* bn, int maxDepth = 5, bool cacheB = false);
    virtual ~StrassenMatrixComputor();

    /*Clear All Command in the Computor*/
    void onReset();

    /*
     It's assume that:
     A is a matrix where each element is a (4,1) vector
     B is a matrix where each element is a (4,4) matrix
     inputs[0] is the transpose of A: AT, inputs[1] is the transpose of B: BT
     outputs[0] is the transpose of C: CT

     Let a be one element of A, b be one element of B,
     then a * b = c is a (4, 1) vector.
     So C is a matrix where each element is a (4,1) vector, the same as A
     */
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    void pushFunction(std::function<void()> function);
    ErrorCode onExecute();

protected:
    Backend* backend() const {
        return mBackend;
    }

private:
    class AddTensor;
    ErrorCode _generateMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT, int currentDepth);
    ErrorCode _generateMatMulConstB(const Tensor* AT, const Tensor* BT, const Tensor* CT, int currentDepth);
    ErrorCode _generateTrivalMatMul(const Tensor* AT, const Tensor* BT, const Tensor* CT);

    std::vector<std::function<void()>> mFunctions;
    std::vector<std::shared_ptr<AddTensor>> mConstTensor;
    int mMaxDepth;
    bool mCacheB;

    Backend* mBackend;
};
} // namespace MNN

#endif /* StrassenMatmulComputor_hpp */
