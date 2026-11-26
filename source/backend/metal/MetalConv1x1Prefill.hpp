#ifndef MNN_METAL_CONV1X1_PREFILL_HPP
#define MNN_METAL_CONV1X1_PREFILL_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "MetalConv1x1Kernels.hpp"

namespace MNN {

struct PrefillCaps {
    bool allowTensorApi = false;
    bool hasSgMatrix = false;
    bool hasSgReduce = false;
    bool lowMemoryKernelsAvailable = false;
    int gpuGen = 0;
    size_t maxThreadgroupMemoryBytes = 0;
};

struct QuantPrefillShape {
    int area = 0;
    int ic = 0;
    int oc = 0;
    int blockCount = 1;
    int dequantBits = 0;
    bool fp16Storage = false;
    bool siluOffer = false;
    bool dualOffer = false;
    int gateUpDualMinArea = 0;
};

enum class PrefillKernel {
    Unsupported,
    TensorInKernel8x64,
    TensorInKernel32x64,
    TensorInKernel64x64,
    TensorRowDeviceM64,
    TensorPackedM32,
    SgReduceG4mN,
    SgReduceG16,
    SgReduceG8,
    SgInKernel8x16,
    SgInKernel32x64,
    SgInKernel32x16,
    SgInKernel16x32,
    SgInKernel16x16,
    SgPreDequantM32,
    SgPreDequantM64,
    ScalarW4,
    ScalarW8
};

enum class PrefillFusion { None, SiluMul, GateUpDual };
enum class WeightPreparation { None, Packed, RowDevice };

struct QuantPrefillPlan {
    PrefillKernel kernel = PrefillKernel::Unsupported;
    int kPartitions = 1;
    bool doubleBuf = false;
    PrefillFusion fusion = PrefillFusion::None;
    int reduceRowsPerPiece = 0;
    int pieceCount = 1;
    bool protectPieceTail = false;
    bool loopK64 = false;
    bool hoistW4Scale = false;

    bool usesTensorApi() const {
        switch (kernel) {
            case PrefillKernel::TensorInKernel8x64:
            case PrefillKernel::TensorInKernel32x64:
            case PrefillKernel::TensorInKernel64x64:
            case PrefillKernel::TensorRowDeviceM64:
            case PrefillKernel::TensorPackedM32:
                return true;
            default:
                return false;
        }
    }

    WeightPreparation weightPreparation() const {
        switch (kernel) {
            case PrefillKernel::TensorRowDeviceM64:
                return WeightPreparation::RowDevice;
            case PrefillKernel::TensorPackedM32:
            case PrefillKernel::SgPreDequantM32:
            case PrefillKernel::SgPreDequantM64:
                return WeightPreparation::Packed;
            default:
                return WeightPreparation::None;
        }
    }

    bool dequantInKernel() const {
        return kernel != PrefillKernel::Unsupported && weightPreparation() == WeightPreparation::None;
    }

    bool usesGemmBindings() const {
        return usesTensorApi() || weightPreparation() != WeightPreparation::None;
    }
};

inline int prefillUpDiv(int x, int y) {
    return (x + y - 1) / y;
}

inline bool prefillBlockFitsQuads(int ic, int blockCount, int quads) {
    return blockCount <= 1 || prefillUpDiv(prefillUpDiv(ic, 4), blockCount) % quads == 0;
}

inline QuantPrefillPlan selectSimdInKernelPrefill(const PrefillCaps& c, const QuantPrefillShape& s) {
    QuantPrefillPlan p;
    const int icDiv4 = prefillUpDiv(s.ic, 4);
    const bool lowBits = s.dequantBits == 2 || s.dequantBits == 3;
    const bool gemvAllArea = lowBits || (c.hasSgMatrix && !prefillBlockFitsQuads(s.ic, s.blockCount, 2));
    if (c.hasSgReduce && (s.area <= 16 || gemvAllArea)) {
        const float ratio = 1.0 * icDiv4 / 2048.0 * s.oc / 2048.0;
        const bool heavyMemory = ratio > 1.0;
        if (s.area <= 16 || (!heavyMemory && s.area <= 32)) {
            p.kernel = PrefillKernel::SgReduceG4mN;
            p.reduceRowsPerPiece = s.area;
            if (s.area > 5 && !heavyMemory) {
                p.reduceRowsPerPiece = prefillUpDiv(s.area, 2);
                p.pieceCount = 2;
                p.protectPieceTail = s.area % 2 != 0;
            }
        } else if (s.oc > 16384 && prefillUpDiv(s.oc, 4) % 2 == 0) {
            p.kernel = PrefillKernel::SgReduceG16;
        } else {
            p.kernel = PrefillKernel::SgReduceG8;
        }
    } else if (c.hasSgMatrix && s.area > 16 && s.oc > 8 && icDiv4 % 2 == 0) {
        if (icDiv4 % 8 != 0 || !prefillBlockFitsQuads(s.ic, s.blockCount, 8)) {
            p.kernel = PrefillKernel::SgInKernel8x16;
        } else if (s.area >= 64 && s.area < 128 && s.oc >= 512 &&
                   (size_t)icDiv4 * s.oc <= 1024 * 1024 && c.maxThreadgroupMemoryBytes >= 8192) {
            p.kernel = PrefillKernel::SgInKernel32x64;
        } else if (s.area >= 128 && s.oc >= 512 && s.area * s.oc > 512 * 2048 &&
                   c.maxThreadgroupMemoryBytes >= 8192) {
            p.kernel = PrefillKernel::SgInKernel32x64;
        } else if (s.area >= 32 && s.area * s.oc > 128 * 2048) {
            p.kernel = PrefillKernel::SgInKernel32x16;
        } else if (s.oc > 512 && s.area * s.oc > 128 * 2048) {
            p.kernel = PrefillKernel::SgInKernel16x32;
        } else {
            p.kernel = PrefillKernel::SgInKernel16x16;
        }
    } else if (s.dequantBits == 4) {
        p.kernel = PrefillKernel::ScalarW4;
    } else if (s.dequantBits == 8) {
        p.kernel = PrefillKernel::ScalarW8;
    }
    return p;
}

inline QuantPrefillPlan selectTensorPrefill(const PrefillCaps& c, const QuantPrefillShape& s) {
    const bool lowBits = s.dequantBits == 2 || s.dequantBits == 3;
    if (c.lowMemoryKernelsAvailable &&
        (!prefillBlockFitsQuads(s.ic, s.blockCount, 8) ||
         (lowBits ? !c.hasSgMatrix : (s.area <= 5 && c.hasSgReduce)))) {
        return selectSimdInKernelPrefill(c, s);
    }
    QuantPrefillPlan p;
    if (c.gpuGen >= 13 && c.gpuGen < 16 && !lowBits && s.area >= 64 && s.ic % 16 == 0 &&
        prefillBlockFitsQuads(s.ic, s.blockCount, 4)) {
        p.kernel = PrefillKernel::TensorRowDeviceM64;
    } else if (s.dequantBits == 3 && (int64_t)s.ic * s.oc < 4 * 1024 * 1024) {
        p.kernel = PrefillKernel::TensorPackedM32;
        p.loopK64 = s.ic > s.oc && s.ic > 2048 && (s.ic / s.blockCount) % 64 == 0;
    } else {
        if (!lowBits && s.area <= 8) {
            p.kernel = PrefillKernel::TensorInKernel8x64;
        } else if (!lowBits && s.area >= 64) {
            p.kernel = PrefillKernel::TensorInKernel64x64;
            p.doubleBuf = s.fp16Storage;
            // Block-level scale hoist raises live registers across the K loop; it pays on
            // gen<17 but loses occupancy on newer dies, so keep them on the in-loop form.
            p.hoistW4Scale = c.gpuGen < 17;
            if (s.area >= s.gateUpDualMinArea) {
                p.fusion = s.dualOffer ? PrefillFusion::GateUpDual :
                           (s.siluOffer ? PrefillFusion::SiluMul : PrefillFusion::None);
            }
        } else {
            p.kernel = PrefillKernel::TensorInKernel32x64;
        }
        if (s.dequantBits == 4 && s.area <= 32 && prefillUpDiv(s.oc, 64) <= 48 && s.blockCount >= 4) {
            p.kPartitions = 4;
        }
    }
    return p;
}

inline QuantPrefillPlan selectSimdPrefill(const PrefillCaps& c, const QuantPrefillShape& s) {
    const bool outerM64 = c.hasSgMatrix && s.area >= 128;
    if (c.lowMemoryKernelsAvailable) {
        if (!prefillBlockFitsQuads(s.ic, s.blockCount, outerM64 ? 4 : 8)) {
            return selectSimdInKernelPrefill(c, s);
        }
        if (s.dequantBits == 2 || s.dequantBits == 3) {
            if (!c.hasSgMatrix) {
                return selectSimdInKernelPrefill(c, s);
            }
        } else if (s.area < 64 || !c.hasSgMatrix ||
                   (s.area < 128 && s.oc >= 512 && (size_t)s.ic * s.oc <= 4 * 1024 * 1024) ||
                   ((size_t)s.ic * s.oc > 4 * 1024 * 1024 && s.area < 512)) {
            return selectSimdInKernelPrefill(c, s);
        }
    }
    QuantPrefillPlan p;
    p.kernel = outerM64 ? PrefillKernel::SgPreDequantM64 : PrefillKernel::SgPreDequantM32;
    return p;
}

inline QuantPrefillPlan selectQuantPrefill(const PrefillCaps& c, const QuantPrefillShape& s) {
    if (s.area <= 1 || s.ic <= 0 || s.oc <= 0 || s.blockCount <= 0 ||
        (s.dequantBits != 2 && s.dequantBits != 3 && s.dequantBits != 4 && s.dequantBits != 8)) {
        return QuantPrefillPlan();
    }
    return c.allowTensorApi ? selectTensorPrefill(c, s) : selectSimdPrefill(c, s);
}

enum class PrefillFinalization { Fixed, ComputeBestGroupAndLocal, GetGridAndThreadgroup };

struct PrefillSize {
    size_t x = 0, y = 0, z = 0;
    PrefillSize() = default;
    PrefillSize(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
};

struct PrefillPipeline {
    std::string entry;
    PrefillSource source = PrefillSource::None;
    std::vector<std::string> keys;
    std::vector<std::pair<std::string, std::string>> macros;
    PrefillSize grid;
    PrefillSize threads;
    PrefillFinalization finalization = PrefillFinalization::Fixed;
    bool optionalCompilation = false;
    bool precompiledFp16 = false;

    const Conv1x1KernelRecord& setKernel(Conv1x1Kernel kernel) {
        const auto& body = conv1x1Kernel(kernel);
        entry = body.entry;
        source = body.source;
        return body;
    }

    void define(const std::string& name, const std::string& value = "1") {
        macros.emplace_back(name, value);
    }
    void specialize(const std::string& name) {
        keys.emplace_back(name);
        define(name);
    }
};

struct PrefillDescription {
    PrefillPipeline main, dequant, pack, reducer;
    std::string profileTag;
    size_t tempWeightBytes = 0;
    size_t packedInputBytes = 0;
    size_t partialBytes = 0;
};

inline PrefillDescription describeQuantPrefill(const QuantPrefillPlan& p, const QuantPrefillShape& s) {
    PrefillDescription d;
    if (p.kernel == PrefillKernel::Unsupported) {
        return d;
    }
    const int icDiv4 = prefillUpDiv(s.ic, 4), ocDiv4 = prefillUpDiv(s.oc, 4);
    const std::string type = s.fp16Storage ? "half" : "float";
    const std::string quant = "W_QUANT_" + std::to_string(s.dequantBits);
    PrefillPipeline base;
    base.define("ftype", type);
    base.define("ftype2", type + "2");
    base.define("ftype4", type + "4");
    base.define("ftype2x4", type + "2x4");
    base.define("ftype4x4", type + "4x4");
    base.define("MNN_METAL_FLOAT32_COMPUTER");
    if (s.fp16Storage) {
        base.define("MNN_METAL_FLOAT16_STORAGE");
    }
    base.keys = {type + "4", "MNN_METAL_FLOAT32_COMPUTER"};
    const bool row = p.weightPreparation() == WeightPreparation::RowDevice;
    if (p.usesGemmBindings()) {
        base.keys.emplace_back("conv1x1_dequant_weight_outter");
        base.optionalCompilation = p.usesTensorApi();
        if (row) {
            base.keys.emplace_back("DEVICE_TENSOR_GEMM");
            d.pack = base;
            const auto& pack = d.pack.setKernel(Conv1x1Kernel::PackInputTensor);
            d.pack.keys.emplace_back(d.pack.entry);
            d.pack.define("USE_METAL_TENSOR_OPS");
            // Mirrors the production launch UP_DIV(input->elementSize()/4, 128); prepare
            // recomputes it from the actual input and saves the dispatch for encode/Replay.
            d.pack.grid = PrefillSize(prefillUpDiv(s.ic * s.area / 4, pack.threads), 1, 1);
            d.pack.threads = PrefillSize(pack.threads, 1, 1);
        }
        if (p.weightPreparation() != WeightPreparation::None) {
            const int bytes = s.fp16Storage ? 2 : 4;
            d.tempWeightBytes = (size_t)ocDiv4 * 4 * prefillUpDiv(s.ic, 32) * 32 * bytes;
            if (row) {
                d.packedInputBytes = (size_t)s.area * icDiv4 * 4 * bytes;
            }
            d.dequant = base;
            d.dequant.setKernel(row ? Conv1x1Kernel::WDequantRow : Conv1x1Kernel::WDequant);
            // RowDevice already has a distinct base key; preserve the shared family key.
            d.dequant.keys.emplace_back(conv1x1Kernel(Conv1x1Kernel::WDequant).entry);
            d.dequant.specialize(quant);
            if (s.ic % 16 != 0) {
                d.dequant.specialize("W_ALIGN_K16_PROTECT");
            }
            if (row) {
                d.dequant.define("USE_METAL_TENSOR_OPS");
            }
            d.dequant.grid = PrefillSize(s.oc, prefillUpDiv(s.ic, 16), 1);
            d.dequant.finalization = PrefillFinalization::ComputeBestGroupAndLocal;
        }
    } else {
        base.define(quant);
        base.keys.emplace_back("conv1x1_wquant_" + std::to_string(s.dequantBits));
    }
    d.main = base;
    Conv1x1Kernel mainKernel = Conv1x1Kernel::Count;
    switch (p.kernel) {
        case PrefillKernel::TensorInKernel8x64:
            mainKernel = s.dequantBits == 8 ? Conv1x1Kernel::FusedQ8StageM8 :
                (p.kPartitions > 1 ? Conv1x1Kernel::FusedQ4StageKSplitM8 : Conv1x1Kernel::FusedQ4StageM8);
            break;
        case PrefillKernel::TensorInKernel32x64:
            mainKernel = p.kPartitions > 1 ? Conv1x1Kernel::FusedQ4StageKSplit : Conv1x1Kernel::FusedQ4Stage;
            break;
        case PrefillKernel::TensorInKernel64x64:
            mainKernel = Conv1x1Kernel::FusedQ4StageM64;
            break;
        case PrefillKernel::TensorRowDeviceM64:
            mainKernel = Conv1x1Kernel::GemmDeviceTensor;
            break;
        case PrefillKernel::TensorPackedM32:
        case PrefillKernel::SgPreDequantM32:
            mainKernel = Conv1x1Kernel::Gemm32x64SplitKSg;
            break;
        case PrefillKernel::SgPreDequantM64:
            mainKernel = Conv1x1Kernel::Gemm64x64SplitKSg;
            break;
        case PrefillKernel::SgReduceG4mN:
        case PrefillKernel::SgReduceG16:
        case PrefillKernel::SgReduceG8:
            d.main.keys.emplace_back("conv1x1_wquant_sg_reduce");
            if (p.kernel == PrefillKernel::SgReduceG4mN) {
                if (p.protectPieceTail) {
                    d.main.specialize("MNN_METAL_SRC_PROTECT");
                }
                d.main.source = kConv1x1GemvG4mSource;
                d.main.entry = conv1x1GemvG4mName(p.reduceRowsPerPiece);
                d.main.grid = PrefillSize(prefillUpDiv(s.oc, kConv1x1GemvG4mTileN), p.pieceCount, 1);
                d.main.threads = PrefillSize(kConv1x1GemvG4mThreads, 1, 1);
            } else {
                const auto& body = d.main.setKernel(p.kernel == PrefillKernel::SgReduceG16
                    ? Conv1x1Kernel::GemvG16 : Conv1x1Kernel::GemvG8);
                d.main.grid = PrefillSize(prefillUpDiv(s.oc, body.tileN), s.area, 1);
                d.main.threads = PrefillSize(body.threads, 1, 1);
            }
            d.main.keys.emplace_back(d.main.entry);
            d.profileTag = d.main.entry;
            return d;
        case PrefillKernel::SgInKernel8x16:
            mainKernel = Conv1x1Kernel::Gemm8x16Wquant;
            break;
        case PrefillKernel::SgInKernel32x64:
            mainKernel = Conv1x1Kernel::Gemm32x64WquantSplitK;
            break;
        case PrefillKernel::SgInKernel32x16:
            mainKernel = Conv1x1Kernel::Gemm32x16Wquant;
            break;
        case PrefillKernel::SgInKernel16x32:
            mainKernel = Conv1x1Kernel::Gemm16x32Wquant;
            break;
        case PrefillKernel::SgInKernel16x16:
            mainKernel = Conv1x1Kernel::Gemm16x16Wquant;
            break;
        case PrefillKernel::ScalarW4:
        case PrefillKernel::ScalarW8: {
            d.main = PrefillPipeline();
            d.main.precompiledFp16 = s.fp16Storage;
            const auto& body = d.main.setKernel(p.kernel == PrefillKernel::ScalarW4
                ? Conv1x1Kernel::ScalarW4 : Conv1x1Kernel::ScalarW8);
            d.main.keys.emplace_back(d.main.entry);
            d.main.grid = PrefillSize(prefillUpDiv(s.area, body.tileM), ocDiv4, 1);
            d.main.finalization = PrefillFinalization::GetGridAndThreadgroup;
            d.profileTag = d.main.entry;
            return d;
        }
        case PrefillKernel::Unsupported:
            return d;
    }
    if (mainKernel == Conv1x1Kernel::Count) {
        return d;
    }
    const auto& body = d.main.setKernel(mainKernel);
    if (!p.usesGemmBindings()) {
        d.main.keys.emplace_back("conv1x1_wquant_sg_matrix");
    }
    d.main.keys.emplace_back(d.main.entry);
    if (p.usesGemmBindings()) {
        if (p.doubleBuf) {
            d.main.keys.emplace_back("FQ_DOUBLE_BUF");
            if (p.kernel == PrefillKernel::TensorInKernel64x64 && s.dequantBits == 4 &&
                s.ic % (s.blockCount * 32) == 0) {
                d.main.specialize("FQ_PREFETCH");
            }
        }
        if (p.fusion == PrefillFusion::GateUpDual) {
            d.main.keys.emplace_back("FQ4_GATEUP_DUAL");
        } else if (p.fusion == PrefillFusion::SiluMul) {
            d.main.keys.emplace_back("FQ4_SILU_MUL");
        }
        if (icDiv4 % 8 != 0) {
            d.main.specialize("MNN_METAL_SRC_PROTECT");
        }
        if (p.usesTensorApi()) {
            d.main.specialize("USE_METAL_TENSOR_OPS");
            // RowDevice ignores LOOP_K64; retain its historical cache key and profile tag.
            if (p.loopK64 || (row && s.ic > s.oc && s.ic > 2048 && (s.ic / s.blockCount) % 64 == 0)) {
                d.main.specialize("LOOP_K64");
            }
        }
        if (p.dequantInKernel()) {
            d.main.specialize(quant);
            if (s.dequantBits == 4) {
                d.main.specialize("FUSED_Q4_REAL_UNPACK");
                if (p.hoistW4Scale && p.kernel == PrefillKernel::TensorInKernel64x64) {
                    d.main.specialize("FQ4_HOIST_BLOCK_SCALE");
                }
            }
        }
        if (p.doubleBuf) {
            d.main.define("FQ_DOUBLE_BUF");
        }
        if (p.fusion == PrefillFusion::SiluMul) {
            d.main.define("FQ4_SILU_MUL");
        } else if (p.fusion == PrefillFusion::GateUpDual) {
            d.main.define("FQ4_GATEUP_DUAL");
        }
    }
    d.profileTag = d.main.keys.back();
    const int tileN = mainKernel == Conv1x1Kernel::FusedQ4StageM64 && p.fusion == PrefillFusion::GateUpDual
        ? body.tileN / 2 : body.tileN;
    d.main.grid = PrefillSize(prefillUpDiv(s.area, body.tileM), prefillUpDiv(s.oc, tileN), p.kPartitions);
    d.main.threads = PrefillSize(body.threads, 1, 1);
    if (p.kPartitions > 1) {
        d.reducer = d.main;
        const auto& reducer = d.reducer.setKernel(Conv1x1Kernel::KSplitReduce);
        d.reducer.keys.emplace_back(d.reducer.entry);
        d.reducer.optionalCompilation = true;
        d.reducer.grid = PrefillSize(prefillUpDiv(ocDiv4 * s.area, reducer.threads), 1, 1);
        d.reducer.threads = PrefillSize(reducer.threads, 1, 1);
        d.partialBytes = (size_t)p.kPartitions * ocDiv4 * s.area * 4 * sizeof(float);
    }
    return d;
}

} // namespace MNN

#endif
