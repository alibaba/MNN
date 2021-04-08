//
//  BF16Binary.cpp
//  MNN
//
//  Created by MNN on 2021/02/07.
//  Copyright © 2021, Alibaba Group Holding Limited
//

#include <algorithm>
#include "backend/cpu/BinaryUtils.hpp"
#include "core/Macro.h"
#include "core/Execution.hpp"
#include "VecHalf.hpp"
#include "math/Vec.hpp"
#include "BF16Backend.hpp"
#include "BF16Functions.hpp"
using Vec4Half = MNN::Math::VecHalf<4>;
using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

class BF16BinaryFloat : public Execution {
public:
    BF16BinaryFloat(Backend *b, int32_t type);
    virtual ~BF16BinaryFloat() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int32_t mType;
    int mNeedBroadcastIndex; // -1 do not need broadcast, 0 for input0, 1 for input1
    int mTotalSize = 0;
};
template<typename Func>
void BF16BinaryWrap(int16_t *dst, const int16_t *src0, const int16_t *src1, const int elementSize, const int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / 4;
    const int remainCount = elementSize - sizeDivUnit * 4;

    float A[4];
    float B[4];
    float C[4];
    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                Vec4::save(A, Vec4(std::move(Vec4Half::load(src0Ptr).value)));
                Vec4::save(B, Vec4(std::move(Vec4Half::load(src1Ptr).value)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
                src0 += 4;
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc0[4];
            int16_t tempSrc1[4];
            int16_t tempDst[4];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
            Vec4::save(A, Vec4(std::move(Vec4Half::load(tempSrc0).value)));
            Vec4::save(B, Vec4(std::move(Vec4Half::load(tempSrc1).value)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(A[v], B[v]);
            }
            Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    } else if (0 == needBroadcastIndex) {
        const int16_t srcValue016 = src0[0];
        float srcValue0;
        BF16Functions::get()->MNNLowpToFp32(&srcValue016, &srcValue0, 1);
        auto a = Vec4Half(srcValue0);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                Vec4::save(B, Vec4(std::move(Vec4Half::load(src1Ptr).value)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc1[4];
            int16_t tempDst[4];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
            Vec4::save(B, Vec4(std::move(Vec4Half::load(tempSrc1).value)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(srcValue0, B[v]);
            }
            Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    } else {
        const int16_t srcValue116 = src1[0];
        float srcValue1;
        BF16Functions::get()->MNNLowpToFp32(&srcValue116, &srcValue1, 1);
        auto b = Vec4Half(srcValue1);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                Vec4::save(A, Vec4(std::move(Vec4Half::load(src0Ptr).value)));
                for (int v = 0; v < 4; ++ v) {
                    C[v] = compute(A[v], B[v]);
                }
                Vec4Half::save(dstPtr, Vec4Half(std::move(Vec4::load(C).value)));
                src0 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc0[4];
            int16_t tempDst[4];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
            Vec4::save(A, Vec4(std::move(Vec4Half::load(tempSrc0).value)));
            for (int v = 0; v < remainCount; ++ v) {
                C[v] = compute(A[v], srcValue1);
            }
            Vec4Half::save(tempDst, Vec4Half(std::move(Vec4::load(C).value)));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    }
}


template<typename Func>
void BF16Binary(int16_t *dst, const int16_t *src0, const int16_t *src1, const int elementSize, const int needBroadcastIndex) {
    Func compute;
    const int sizeDivUnit = elementSize / 4;
    const int remainCount = elementSize - sizeDivUnit * 4;

    if (-1 == needBroadcastIndex) {
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                Vec4Half a = Vec4Half::load(src0);
                Vec4Half b = Vec4Half::load(src1);
                Vec4Half::save(dst, compute(a, b));
                src0 += 4;
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc0[4];
            int16_t tempSrc1[4];
            int16_t tempDst[4];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
            ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
            Vec4Half a = Vec4Half::load(tempSrc0);
            Vec4Half b = Vec4Half::load(tempSrc1);
            Vec4Half::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    } else if (0 == needBroadcastIndex) {
        const int16_t srcValue016 = src0[0];
        float srcValue0;
        BF16Functions::get()->MNNLowpToFp32(&srcValue016, &srcValue0, 1);
        Vec4Half a = Vec4Half(srcValue0);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src1Ptr = src1;
                auto dstPtr = dst;
                Vec4Half b = Vec4Half::load(src1Ptr);
                Vec4Half::save(dstPtr, compute(a, b));
                src1 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc1[8];
            int16_t tempDst[8];
            ::memcpy(tempSrc1, src1, remainCount * sizeof(int16_t));
            Vec4Half b = Vec4Half::load(tempSrc1);
            Vec4Half::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    } else {
        const int16_t srcValue116 = src1[0];
        float srcValue1;
        BF16Functions::get()->MNNLowpToFp32(&srcValue116, &srcValue1, 1);
        Vec4Half b = Vec4Half(srcValue1);
        if (sizeDivUnit > 0) {
            for (int i = 0; i < sizeDivUnit; ++i) {
                const auto src0Ptr = src0;
                auto dstPtr = dst;
                Vec4Half a = Vec4Half::load(src0Ptr);
                Vec4Half::save(dstPtr, compute(a, b));
                src0 += 4;
                dst += 4;
            }
        }
        if (remainCount > 0) {
            int16_t tempSrc0[8];
            int16_t tempDst[8];
            ::memcpy(tempSrc0, src0, remainCount * sizeof(int16_t));
            Vec4Half a = Vec4Half::load(tempSrc0);
            Vec4Half::save(tempDst, compute(a, b));
            ::memcpy(dst, tempDst, remainCount * sizeof(int16_t));
        }
    }
}


struct VecBinaryAdd : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return x + y;
    }
};

struct VecBinarySub : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return x - y;
    }
};

struct VecBinaryMul : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return x * y;
    }
};

struct VecBinaryMin : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return Vec4Half::min(x, y);
    }
};

struct VecBinaryMax : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return Vec4Half::max(x, y);
    }
};

struct VecBinarySqd : std::binary_function<Vec4Half, Vec4Half, Vec4Half> {
    Vec4Half operator()(const Vec4Half& x, const Vec4Half& y) const {
        return (x-y)*(x-y);
    }
};

BF16BinaryFloat::BF16BinaryFloat(Backend *backend, int32_t type):Execution(backend), mType(type) {
    // Do nothing
}

ErrorCode BF16BinaryFloat::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == outputs.size());
    const int input0DataCount = inputs[0]->elementSize();
    const int input1DataCount = inputs[1]->elementSize();
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
        mTotalSize = input1DataCount;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
        mTotalSize = input1DataCount;
    } else {
        mNeedBroadcastIndex = 1;
        mTotalSize = input0DataCount;
    }
    return NO_ERROR;
}

ErrorCode BF16BinaryFloat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs){
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    const auto src0 = input0->host<int16_t>();
    const auto src1 = input1->host<int16_t>();
    auto dst = output->host<int16_t>();
    
    switch (mType) {
        case BinaryOpOperation_ADD:
            BF16Binary<VecBinaryAdd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SUB:
            BF16Binary<VecBinarySub>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MUL:
            BF16Binary<VecBinaryMul>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MINIMUM:
            BF16Binary<VecBinaryMin>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MAXIMUM:
            BF16Binary<VecBinaryMax>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_SquaredDifference:
            BF16Binary<VecBinarySqd>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_REALDIV:
            BF16BinaryWrap<BinaryRealDiv<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_FLOORDIV:
            BF16BinaryWrap<BinaryFloorDiv<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_FLOORMOD:
            BF16BinaryWrap<BinaryFloorMod<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_POW:
            BF16BinaryWrap<BinaryPow<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_ATAN2:
            BF16BinaryWrap<BinaryAtan2<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        case BinaryOpOperation_MOD:
            BF16BinaryWrap<BinaryMod<float, float, float>>(dst, src0, src1, mTotalSize, mNeedBroadcastIndex);
            break;
        default:
            return NOT_SUPPORT;
            break;
    }
    return NO_ERROR;
}

class BF16BinaryCreator : public BF16Backend::BF16Creator {
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = outputs[0]->getType();
        if (dataType.code != halide_type_float) {
            return nullptr;
        }
        return new BF16BinaryFloat(backend, type);
    }
};

REGISTER_BF16_OP_CREATOR(OpType_BinaryOp, BF16BinaryCreator);



} // namespace MNN
