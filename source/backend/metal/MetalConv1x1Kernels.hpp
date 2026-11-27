#ifndef MNN_METAL_CONV1X1_KERNELS_HPP
#define MNN_METAL_CONV1X1_KERNELS_HPP

#include <cstddef>
#include <string>

namespace MNN {

enum class PrefillSource { None, WfpMatrix, WqMatrix, WqReduce, Precompiled };

// Entry points used by quantized Conv1x1. Rows own fixed launch geometry; a zero
// denotes a dimension decided by the plan or the backend. Shape-dependent macros
// (quantization, fusion, tails, etc.) still belong to the pipeline description.
#define MNN_CONV1X1_KERNELS(X) \
    X(FusedQuantStageGemm,       "conv1x1_fused_q4_gemm_stage_m64",         WfpMatrix,  64, 64, 128) \
    X(DirectDeviceTensorGemm, "conv1x1_gemm_device_tensor",             WfpMatrix,  64, 64, 128) \
    X(Gemm32x64SplitKSg,     "conv1x1_gemm_32x64_split_k_sg",           WfpMatrix,  32, 64, 128) \
    X(Gemm64x64SplitKSg,     "conv1x1_gemm_64x64_split_k_sg",           WfpMatrix,  64, 64, 128) \
    X(KSplitReduce,          "conv1x1_fused_q4_ksplit_reduce",          WfpMatrix,   0,  0, 128) \
    X(PackInputTensor,       "conv1x1_pack_input_tensor",               WfpMatrix,   0,  0, 128) \
    X(WDequant,              "conv1x1_w_dequant",                       WfpMatrix,   0,  0,   0) \
    X(WDequantRow,           "conv1x1_w_dequant_row",                   WfpMatrix,   0,  0,   0) \
    X(GemvG4m1_2sg,          "conv1x1_gemv_g4m1_2sg_wquant_sg",         WqReduce,    1,  0,   0) \
    X(GemvG8,                "conv1x1_gemv_g8_wquant_sg",               WqReduce,    1,  8, 128) \
    X(GemvG16,               "conv1x1_gemv_g16_wquant_sg",              WqReduce,    1, 16,  64) \
    X(Gemm8x16WquantSplitKFp16, "conv1x1_gemm_8x16_split_k_wquant_sg",   WqMatrix,    8, 16, 192) \
    X(Gemm8x16WquantSplitKFp32, "conv1x1_gemm_8x16_split_k_wquant_sg",   WqMatrix,    8, 32, 256) \
    X(Gemm8x16Wquant,        "conv1x1_gemm_8x16_wquant_sg",             WqMatrix,    8, 16,  32) \
    X(Gemm16x16Wquant,       "conv1x1_gemm_16x16_wquant_sg",            WqMatrix,   16, 16,  32) \
    X(Gemm16x32Wquant,       "conv1x1_gemm_16x32_wquant_sg",            WqMatrix,   16, 32,  32) \
    X(Gemm32x16Wquant,       "conv1x1_gemm_32x16_wquant_sg",            WqMatrix,   32, 16,  32) \
    X(Gemm32x64WquantSplitK, "conv1x1_gemm_32x64_wquant_split_k_sg",    WqMatrix,   32, 64, 128) \
    X(ScalarW4,              "conv1x1_g1z4_w4",                        Precompiled, 4,  4,   0) \
    X(ScalarW8,              "conv1x1_g1z4_w8",                        Precompiled, 4,  4,   0)

enum class Conv1x1Kernel {
#define MNN_CONV1X1_ENUM(id, ...) id,
    MNN_CONV1X1_KERNELS(MNN_CONV1X1_ENUM)
    Count
#undef MNN_CONV1X1_ENUM
};

struct Conv1x1KernelRecord {
    const char* entry;
    PrefillSource source;
    int tileM, tileN, threads;
};

inline const Conv1x1KernelRecord& conv1x1Kernel(Conv1x1Kernel kernel) {
    static constexpr Conv1x1KernelRecord table[] = {
#define MNN_CONV1X1_RECORD(id, entry, source, m, n, threads) {entry, PrefillSource::source, m, n, threads},
        MNN_CONV1X1_KERNELS(MNN_CONV1X1_RECORD)
#undef MNN_CONV1X1_RECORD
    };
    static_assert(sizeof(table) / sizeof(table[0]) == static_cast<std::size_t>(Conv1x1Kernel::Count),
                  "Every Conv1x1Kernel must have a registry row");
    return table[static_cast<int>(kernel)];
}

#undef MNN_CONV1X1_KERNELS

// G4mN publishes N=2..16 under host_name aliases. Decode's one-row body
// (GemvG4m1_2sg) is separate and uses a different launch layout.
constexpr int kConv1x1GemvG4mMin = 2;
constexpr int kConv1x1GemvG4mMax = 16;
constexpr PrefillSource kConv1x1GemvG4mSource = PrefillSource::WqReduce;
constexpr int kConv1x1GemvG4mTileN = 8;
constexpr int kConv1x1GemvG4mThreads = 64;

inline std::string conv1x1GemvG4mName(int rowsPerPiece) {
    return "conv1x1_gemv_g4m" + std::to_string(rowsPerPiece) + "_wquant_sg";
}

} // namespace MNN

#endif // MNN_METAL_CONV1X1_KERNELS_HPP
