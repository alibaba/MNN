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

    // Kernel param accessors. The (const KernelParam&) forms target an explicit kernel slot; the
    // short forms use the primary slot (mParam). The slot forms let the hybrid path query the NEON slot.
    size_t getMr(const KernelParam& p, size_t m) const { return (m == 1) ? p.mKaiMrGemv : p.mKaiMrGemm; }
    size_t getNr(const KernelParam& p) const { return p.mKaiNr; }
    size_t getKr(const KernelParam& p) const { return p.mKaiKr; }
    size_t getSr(const KernelParam& p) const { return p.mKaiSr; }
    size_t getNStep(const KernelParam& p) const { return p.mKaiNStep; }
    size_t getMr(size_t m = 1) const { return getMr(mParam, m); }
    size_t getNr() const { return getNr(mParam); }
    size_t getKr() const { return getKr(mParam); }
    size_t getSr() const { return getSr(mParam); }
    size_t getNStep() const { return getNStep(mParam); }
    bool bSupportSme2() const { return mSme2; }
    static size_t getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep);
    static size_t getDstOffset(size_t mIdx, size_t nIdx, size_t n, size_t elementSize) { return (nIdx * elementSize) + mIdx * (n * elementSize); }

    // Rhs (weight) pack. The (u, p) overloads target an explicit kernel slot; the short forms use
    // the primary slot (mUkernel / mParam).
    size_t getRhsPackedSize(const Ukernel& u, const KernelParam& p, size_t n, size_t k, size_t bl) const;
    size_t getRhsPackedOffset(const Ukernel& u, const KernelParam& p, size_t nIdx, size_t k, size_t bl) const;
    void runRhsPack(const Ukernel& u, const KernelParam& p, size_t numGroups, size_t n, size_t k, size_t bl,
                    const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) const;
    size_t getRhsPackedSize(size_t n, size_t k, size_t bl) const { return getRhsPackedSize(mUkernel, mParam, n, k, bl); }
    size_t getRhsPackedOffset(size_t nIdx, size_t k, size_t bl) const { return getRhsPackedOffset(mUkernel, mParam, nIdx, k, bl); }
    void runRhsPack(size_t numGroups, size_t n, size_t k, size_t bl,
                    const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked) const {
        runRhsPack(mUkernel, mParam, numGroups, n, k, bl, rhs, scale, zeroPoint, bias, rhsPacked);
    }

    // Lhs (activation) dynamic quant + pack.
    size_t getLhsQuantedPackedSize(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl) const;
    size_t getLhsQuantedPackedOffset(const Ukernel& u, const KernelParam& p, size_t m, size_t mIdx, size_t k, size_t bl) const;
    void runLhsQuantPack(const Ukernel& u, const KernelParam& p, size_t m, size_t k, size_t bl, size_t mr,
                         const void* lhs, void* lhsQuantedPacked) const;
    size_t getLhsQuantedPackedSize(size_t m, size_t k, size_t bl) const { return getLhsQuantedPackedSize(mUkernel, mParam, m, k, bl); }
    size_t getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k, size_t bl) const { return getLhsQuantedPackedOffset(mUkernel, mParam, m, mIdx, k, bl); }
    void runLhsQuantPack(size_t m, size_t k, size_t bl, size_t mr, const void* lhs, void* lhsQuantedPacked) const {
        runLhsQuantPack(mUkernel, mParam, m, k, bl, mr, lhs, lhsQuantedPacked);
    }

    // Matmul.
    void runMatmul(const Ukernel& u, const KernelParam& p, size_t m, size_t n, size_t k, size_t bl,
                   const void* lhsPacked, const void* rhsPacked, void* dst,
                   size_t dstStrideRow, size_t dstStrideCol, const float scalarMax, const float scalarMin) const;
    void runMatmul(size_t m, size_t n, size_t k, size_t bl,
                   const void* lhsPacked, const void* rhsPacked, void* dst,
                   size_t dstStrideRow, size_t dstStrideCol, const float scalarMax, const float scalarMin) const {
        runMatmul(mUkernel, mParam, m, n, k, bl, lhsPacked, rhsPacked, dst, dstStrideRow, dstStrideCol, scalarMax, scalarMin);
    }

    std::shared_ptr<Tensor> mWeightInt8;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    // Secondary NEON slot, packed alongside the primary SME slot to enable concurrent SME+NEON
    // execution (see onExecute). Only populated when mHybrid is true.
    std::shared_ptr<Tensor> mWeightInt8Neon;
    std::shared_ptr<Tensor> mTempIm2ColBufferNeon;
    std::shared_ptr<Tensor> mInputConvertBuffer;
    std::shared_ptr<Tensor> mOutputConvertBuffer;
    KernelType mKernelType = KernelType::KERNEL_TYPE_ERROR;
    int32_t mBlockNum = 1;
    bool mSme2 = false;
    bool mDot = false;
    bool mI8mm = false;
    // True for channel-quantized types: the effective block length passed to the ukernels is k.
    bool mChnlQuant = false;
    // True when both the primary SME slot and the secondary NEON slot are configured, enabling the
    // hybrid path that runs an SME kernel and NEON kernels concurrently across threads.
    bool mHybrid = false;
    // True when the active primary slot is the NEON fallback rather than SME2.
    bool mPrimaryIsNeon = false;
    KernelParam mParam;
    Ukernel mUkernel;
    KernelParam mParamNeon;
    Ukernel mUkernelNeon;
};

} // namespace MNN
#endif /* KleidiAIConvInt8_hpp */
