//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KleidiAIConvolution_hpp
#define KleidiAIConvolution_hpp
#include "backend/cpu/CPUConvolution.hpp"
namespace MNN {
class KleidiAIConvolution : public CPUConvolution{
    public:
        enum class KernelType {
            FP16 = 0,
            FP32,
            KERNEL_TYPE_ERROR
        };

        // Whether the running CPU can accelerate a float convolution of this precision.
        static bool isSupported(bool bFP16);

        KleidiAIConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight, size_t originWeightSize, const float *bias, size_t biasSize);
        KleidiAIConvolution(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b);
        virtual ~KleidiAIConvolution();

        virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

        virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
        virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    private:
        // Per-kernel packing parameters resolved from mKernelType and the running CPU.
        struct KernelParam {
            size_t mKaiMstepGemm = 0;
            size_t mKaiNStep = 0;
            size_t mKaiMrGemm = 0;
            size_t mKaiNr = 0;
            size_t mKaiKr = 0;
            size_t mKaiSr = 0;
        };

        // The concrete KleidiAI micro-kernels selected once by configKernel() according to
        // mKernelType. All data pointers are typed as void* so this header stays free of the
        // KleidiAI ukernel headers; dtype (fp16/fp32) is fully encoded by which function is bound.
        struct Ukernel {
            size_t (*rhsPackedSize)(size_t n, size_t k) = nullptr;
            void (*runRhsPack)(size_t numGroups, size_t n, size_t k, size_t nr, size_t kr, size_t sr,
                               size_t rhsStride, const void* rhs, const void* bias, const void* scale,
                               void* rhsPacked, size_t extraBytes, const void* params) = nullptr;
            size_t (*lhsPackedSize)(size_t m, size_t k, size_t mr, size_t kr, size_t sr) = nullptr;
            void (*runLhsPack)(size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t mIdxStart,
                               const void* lhs, size_t lhsStride, void* lhsPacked) = nullptr;
            void (*matmulGemm)(size_t m, size_t n, size_t k, const void* lhsPacked, const void* rhsPacked,
                               void* dst, size_t dstStrideRow, size_t dstStrideCol,
                               float clampMin, float clampMax) = nullptr;
            void (*matmulGemv)(size_t m, size_t n, size_t k, const void* lhsPacked, size_t lhsStride,
                               const void* rhsPacked, void* dst, size_t dstStrideRow, size_t dstStrideCol,
                               float clampMin, float clampMax) = nullptr;
        };

        // Resolve mSme2 from the CPU and fill mParam / mUkernel based on mKernelType.
        void configKernel();

        size_t getMr(size_t m = 1) const { return mParam.mKaiMrGemm; }
        size_t getNr() const { return mParam.mKaiNr; }
        size_t getKr() const { return mParam.mKaiKr; }
        size_t getSr() const { return mParam.mKaiSr; }

        size_t getRhsPackedSize(size_t n, size_t k) const;
        void runRhsPack(size_t numGroups, size_t n, size_t k, size_t rhsStride,
                        const void* rhs, const void* scale, const void* bias, void* rhsPacked) const;
        size_t getLhsPackedSize(size_t m, size_t k) const;
        void runLhsPack(size_t m, size_t k, const void* lhs, size_t lhsStride, void* lhsPacked) const;
        void runMatmul(size_t m, size_t n, size_t k,
                       const void* lhsPacked, const void* rhsPacked, void* dst,
                       size_t dstStrideRow, size_t dstStrideCol,
                       const float scalarMax, const float scalarMin) const;

        std::shared_ptr<Tensor> mInputResource;
        std::shared_ptr<Tensor> mInputConvertBuffer;
        std::shared_ptr<Tensor> mOutputConvertBuffer;
        std::shared_ptr<CPUConvolution::Resource> mResource;
        KernelType mKernelType = KernelType::KERNEL_TYPE_ERROR;
        bool mSme2 = false;
        KernelParam mParam;
        Ukernel mUkernel;
        size_t mElementSize = 0;
        std::vector<float> mPostParameters;
};
} // namespace MNN
#endif
