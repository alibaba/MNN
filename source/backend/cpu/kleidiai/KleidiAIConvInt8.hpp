//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KleidiAIConvInt8_hpp
#define KleidiAIConvInt8_hpp
#include "backend/cpu/CPUConvolution.hpp"

namespace MNN {
class KleidiAIConvInt8 : public CPUConvolution {
public:
    // Quantized acceleration type. Encodes bit width / symmetry / quant granularity / activation type.
    enum class KernelType {
        QI4_ASYM_PERCHANNEL_F32 = 0,
        QI4_ASYM_PERCHANNEL_F16,
        QI4_ASYM_PERBLOCK_F32,
        QI4_ASYM_PERBLOCK_F16,
        QI4_SYM_PERCHANNEL_F32,
        QI4_SYM_PERBLOCK,
        QI8_ASYM_PERCHANNEL,
        QI8_ASYM_PERBLOCK,
        QI8_SYM_PERCHANNEL,
        QI8_SYM_PERBLOCK,
        KERNEL_TYPE_ERROR
    };

    // Classify the quantized weight into a KernelType (was KleidiAI::getQIntAccelType).
    static KernelType getKernelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes);
    // Whether the current CPU + convolution shape can be accelerated by KleidiAI.
    static bool isSupported(KernelType type, const Convolution2DCommon* common);

    KleidiAIConvInt8(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant, KernelType kernelType, int32_t blockNum);
    virtual ~KleidiAIConvInt8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    KleidiAIConvInt8(Backend* backend, const Op* op, const KleidiAIConvInt8& exe);

    // Per-kernel packing parameters resolved from mKernelType and the running CPU.
    struct KernelParam {
        size_t mKaiMstepGemv = 0;
        size_t mKaiMstepGemm = 0;
        size_t mKaiNStep = 0;
        size_t mKaiMrGemv = 0;
        size_t mKaiMrGemm = 0;
        size_t mKaiNr = 0;
        size_t mKaiKr = 0;
        size_t mKaiSr = 0;
    };

    // The concrete KleidiAI micro-kernels selected once by configKernel() according to
    // mKernelType + the running CPU. Every data pointer is void* and every entry uses a single
    // uniform signature (bl is ignored by kernels that do not take it), so this header stays free
    // of the KleidiAI ukernel headers. The .cpp binds each slot to a thin adapter that forwards to
    // the concrete kernel, absorbing the per-family signature differences (bl / params / helpers).
    struct Ukernel {
        size_t (*rhsPackedSize)(size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) = nullptr;
        size_t (*rhsPackedOffset)(size_t nIdx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) = nullptr;
        void (*runRhsPack)(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl,
                           const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                           void* rhsPacked) = nullptr;
        size_t (*lhsPackedSize)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) = nullptr;
        size_t (*lhsPackedOffset)(size_t mIdx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) = nullptr;
        void (*runLhsQuantPack)(size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr,
                                const void* lhs, void* lhsQuantedPacked) = nullptr;
        // Handles both GEMV (m == 1) and GEMM internally.
        void (*matmul)(size_t m, size_t n, size_t k, size_t bl,
                       const void* lhsPacked, const void* rhsPacked, void* dst,
                       size_t dstStrideRow, size_t dstStrideCol, float clampMin, float clampMax) = nullptr;
    };

    // Resolve mSme2/mDot/mI8mm from the CPU and fill mParam / mUkernel based on mKernelType.
    void configKernel();

    // Kernel param accessors.
    size_t getMr(size_t m = 1) const { return (m == 1) ? mParam.mKaiMrGemv : mParam.mKaiMrGemm; }
    size_t getNr() const { return mParam.mKaiNr; }
    size_t getKr() const { return mParam.mKaiKr; }
    size_t getSr() const { return mParam.mKaiSr; }
    size_t getNStep() const { return mParam.mKaiNStep; }
    bool bSupportSme2() const { return mSme2; }
    static size_t getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep);
    static size_t getDstOffset(size_t mIdx, size_t nIdx, size_t n, size_t elementSize) { return (nIdx * elementSize) + mIdx * (n * elementSize); }

    // Rhs (weight) pack.
    size_t getRhsPackedSize(size_t n, size_t k, size_t bl) const;
    size_t getRhsPackedOffset(size_t nIdx, size_t k, size_t bl) const;
    void runRhsPack(size_t numGroups, size_t n, size_t k, size_t bl,
                    const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                    void* rhsPacked) const;

    // Lhs (activation) dynamic quant + pack.
    size_t getLhsQuantedPackedSize(size_t m, size_t k, size_t bl) const;
    size_t getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k, size_t bl) const;
    void runLhsQuantPack(size_t m, size_t k, size_t bl, size_t mr, const void* lhs, void* lhsQuantedPacked) const;

    // Matmul.
    void runMatmul(size_t m, size_t n, size_t k, size_t bl,
                   const void* lhsPacked, const void* rhsPacked, void* dst,
                   size_t dstStrideRow, size_t dstStrideCol,
                   const float scalarMax, const float scalarMin) const;

    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<Tensor> mInputConvertBuffer;
    std::shared_ptr<Tensor> mOutputConvertBuffer;
    KernelType mKernelType = KernelType::KERNEL_TYPE_ERROR;
    int32_t mBlockNum = 1;
    bool mSme2 = false;
    bool mDot = false;
    bool mI8mm = false;
    // True for channel-quantized types: the effective block length passed to the ukernels is k.
    bool mChnlQuant = false;
    KernelParam mParam;
    Ukernel mUkernel;
};

} // namespace MNN
#endif /* KleidiAIConvInt8_hpp */
