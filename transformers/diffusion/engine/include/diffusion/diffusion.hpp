//
//  diffusion.hpp
//
//  Created by MNN on 2025/01/12.
//  MNN
//
#ifndef MNN_DIFFUSION_HPP
#define MNN_DIFFUSION_HPP

#include <map>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <functional>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/MathOp.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN;
using namespace MNN::Express;

namespace MNN {
namespace DIFFUSION {

// Philox RNG implementation (aligned with PyTorch)
// Based on: Salmon et al. 2011, "Parallel Random Numbers: As Easy as 1, 2, 3"
class PhiloxRNG {
public:
    using result_type = uint32_t;
    using key_type = std::array<uint32_t, 2>;
    using counter_type = std::array<uint32_t, 4>;
    
    static constexpr uint32_t PHILOX_M4x32_0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
    static constexpr uint32_t PHILOX_W32_0 = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W32_1 = 0xBB67AE85;
    
    PhiloxRNG(uint64_t seed = 0, uint64_t offset = 0) {
        key_[0] = static_cast<uint32_t>(seed);
        key_[1] = static_cast<uint32_t>(seed >> 32);
        counter_[0] = static_cast<uint32_t>(offset);
        counter_[1] = static_cast<uint32_t>(offset >> 32);
        counter_[2] = 0;
        counter_[3] = 0;
        index_ = 0;
    }
    
    uint32_t operator()() {
        if (index_ == 0) {
            counter_type result = philox4x32_10(counter_, key_);
            output_[0] = result[0]; output_[1] = result[1];
            output_[2] = result[2]; output_[3] = result[3];
            increment_counter();
        }
        uint32_t ret = output_[index_];
        index_ = (index_ + 1) & 3;
        return ret;
    }
    
    float uniform() {
        uint32_t x = (*this)();
        return static_cast<float>(x) * (1.0f / 4294967296.0f);
    }
    
    float randn() {
        float u1 = uniform();
        float u2 = uniform();
        u1 = u1 < 1e-10f ? 1e-10f : u1;
        float r = std::sqrt(-2.0f * std::log(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        return r * std::cos(theta);
    }
    
private:
    key_type key_;
    counter_type counter_;
    uint32_t output_[4];
    int index_;
    
    void increment_counter() {
        counter_[0]++;
        if (counter_[0] == 0) { counter_[1]++;
            if (counter_[1] == 0) { counter_[2]++;
                if (counter_[2] == 0) { counter_[3]++; }
            }
        }
    }
    
    static inline uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hi) {
        uint64_t product = static_cast<uint64_t>(a) * b;
        *hi = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
    }
    
    static counter_type philox4x32_round(counter_type ctr, key_type key) {
        uint32_t hi0, hi1;
        uint32_t lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0], &hi0);
        uint32_t lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2], &hi1);
        counter_type ret;
        ret[0] = hi1 ^ ctr[1] ^ key[0]; ret[1] = lo1;
        ret[2] = hi0 ^ ctr[3] ^ key[1]; ret[3] = lo0;
        return ret;
    }
    
    static counter_type philox4x32_10(counter_type ctr, key_type key) {
        for (int i = 0; i < 10; ++i) {
            ctr = philox4x32_round(ctr, key);
            if (i < 9) { key[0] += PHILOX_W32_0; key[1] += PHILOX_W32_1; }
        }
        return ctr;
    }
};

class Tokenizer;
typedef enum {
    STABLE_DIFFUSION_1_5 = 0,
    STABLE_DIFFUSION_TAIYI_CHINESE = 1,
    SANA_DIFFUSION = 2,
    STABLE_DIFFUSION_ZIMAGE = 3,
    LONGCAT_IMAGE_EDIT = 4,
    DIFFUSION_MODEL_USER
} DiffusionModelType;

// GPU memory mode for OpenCL backend
typedef enum {
    GPU_MEMORY_AUTO = 0,
    GPU_MEMORY_BUFFER = 1,
    GPU_MEMORY_IMAGE = 2,
} DiffusionGpuMemoryMode;

// Precision mode for inference
typedef enum {
    PRECISION_AUTO = 0,
    PRECISION_LOW = 1,
    PRECISION_NORMAL = 2,
    PRECISION_HIGH = 3,
} DiffusionPrecisionMode;

// CFG mode for dual-UNet models (e.g., LongCat)
typedef enum {
    CFG_MODE_AUTO = 0,
    CFG_MODE_WIDE = 1,
    CFG_MODE_STANDARD = 2,
    CFG_MODE_MEDIUM = 3,
    CFG_MODE_NARROW = 4,
    CFG_MODE_MINIMAL = 5,
} DiffusionCFGMode;

// DiffusionConfig is defined in diffusion_config.hpp (uses rapidjson, not included here to keep header lightweight)

// LLM Text Encoder configuration for multimodal models (e.g., LongCat)
struct LLMEncoderConfig {
    int prefixLen = 67;
    int suffixLen = 5;
    int targetSeqLen = 838;
    int tokenizerMaxLength = 512;
    int visionResizeSize = 512;
    int hiddenSize = 3584;
    
    static LLMEncoderConfig longcat() {
        LLMEncoderConfig cfg;
        return cfg;
    }
};

// Scheduler type for unified scheduler API
enum SchedulerType {
    SCHEDULER_PLMS,
    SCHEDULER_EULER
};

// UNet output preprocessing function type
using UNetPreprocessFunc = std::function<VARP(VARP)>;

class MNN_PUBLIC Diffusion {
public:
    Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual ~Diffusion();
    
    // Factory methods
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    // Extended factory with all options for ZImage/LongCat
    static Diffusion* createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads = 4);
    
    // Official run interface (text prompt)
    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) = 0;
    // Extended run interface with cfgScale and inputImagePath
    virtual bool run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath = "");

    // Official unified interface (Sana-style with input_embeds)
    virtual bool run(const VARP input_embeds, 
                    const std::string& mode,
                    const std::string& inputImagePath,
                    const std::string& outputImagePath,
                    int width, int height,
                    int iterNum, int randomSeed,
                    bool use_cfg, float cfg_scale,
                    std::function<void(int)> progressCallback) = 0;
    
    virtual bool load() = 0;

    // Image processing utility functions (static, model-agnostic)
    static VARP resizeAndCenterCrop(VARP image, int targetW, int targetH);
    static VARP bgrToRgb(VARP bgrImage);
    static VARP rgbToBgr(VARP rgbImage);
    static VARP hwcToNchw(VARP hwcImage, bool normalize = false);
    static VARP nchwToHwc(VARP nchwImage, bool denormalize = false);
    
    // Latent packing/unpacking for Flux-like models
    static void packLatents(const float* src, float* dst, int B, int C, int H, int W, int seqOffset = 0);
    static void unpackLatents(const float* src, float* dst, int B, int C, int H, int W, int seqLen);

protected:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_;
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_cpu_;
    std::vector<std::shared_ptr<Module>> mModules;
    
    std::string mModelPath;
    DiffusionModelType mModelType;
    /* 0 -> memory saving mode, for memory stictly limited application
        1 -> memory enough mode, for better image generation speed
        2 -> balance mode for memory and generation speed.
     */
    int mMemoryMode;
    MNNForwardType mBackendType;
    bool mTextEncoderOnCPU = true;
    bool mVaeOnCPU = false;
    DiffusionGpuMemoryMode mGpuMemoryMode = GPU_MEMORY_AUTO;
    DiffusionPrecisionMode mPrecisionMode = PRECISION_AUTO;
    DiffusionCFGMode mCFGMode = CFG_MODE_AUTO;
    int mNumThreads = 4;
    
    int mImageWidth = 0;
    int mImageHeight = 0;
};

}
}
#endif
