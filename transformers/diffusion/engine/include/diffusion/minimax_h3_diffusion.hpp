//
//  minimax_h3_diffusion.hpp
//
//  MiniMax-H3 joint video + audio generation runtime.
//
#ifndef MNN_MINIMAX_H3_DIFFUSION_HPP
#define MNN_MINIMAX_H3_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>
#include <string>
#include <vector>

namespace MNN {
namespace DIFFUSION {

// MiniMax-H3 runs one stack of blocks over a single packed sequence holding the text condition, the keyframe
// conditioning rows, the audio rows and the target video rows. Attention over it is full and non-causal, and
// the checkpoint is guidance-distilled, so there is no unconditional pass and no CFG.
//
// Two properties of the checkpoint shape this runtime:
//
//   * Every AdaLN projection reads only the timestep embedding, so for a fixed schedule the whole 13B branch
//     is constant. It is folded offline into a per-(step, layer) table, which is why the device never sees
//     26 GB of AdaLN weights.
//   * The 50 blocks are exported as several sequential partitions so no single graph has to be built, and on
//     NPU backends so no single context binary has to be finalized, for a 20B stack.
class MNN_PUBLIC MinimaxH3Diffusion : public Diffusion {
public:
    MinimaxH3Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType,
                       int memoryMode);
    virtual ~MinimaxH3Diffusion();

    virtual bool load() override;

    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback) override;

    virtual bool run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                     const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                     bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) override;

    bool runVideo(const std::string& prompt, const std::string& outputDir, int width, int height, int frames,
                  int steps, int seed, float cfgScale, std::function<void(int)> progressCallback) override;

    // Runs the denoising loop from precomputed conditioning and writes the video and audio latents.
    // `promptEmbedsPath` holds `numTextTokens * textDim` little-endian floats -- the H3 encoder's hidden
    // states. The first release keeps the encoder out of the device pipeline, so conditioning arrives as a
    // tensor rather than as text.
    bool runFromPromptEmbeds(const std::string& promptEmbedsPath, const std::string& outputDir, int seed,
                             std::function<void(int)> progressCallback);

    // Geometry the resources were exported for. A partition's shapes are static, so a request has to match.
    int sequenceLength() const;
    int numVideoRows() const;
    int numAudioRows() const;
    int numSteps() const;
    int numConditionVideoRows() const;

    // Caps how many block partitions stay resident, reloading the rest as the stack advances.
    //
    // Backends with native low-bit kernels want every partition resident, which is the default (0). MNN's
    // CUDA backend instead materializes int4 weights as fp16 -- about 810 MiB per layer -- so a 50-layer
    // stack needs roughly 40 GB there and a window is what makes it runnable on a 24 GB device. The reloads
    // come out of the page cache, not from storage, but they are not free.
    void setResidentGroups(int groups);

    // Activation precision of every module.
    //
    // H3's residual stream carries outliers far outside float16's range -- a block-0 output already reaches
    // absmax ~3.5e4 against a 6.55e4 ceiling, and it grows over 50 blocks -- so float16 activations overflow
    // to NaN. That is why the released checkpoint is bfloat16. Prefer `Precision_Low_BF16` where the backend
    // has it (MNN CUDA needs `MNN_CUDA_BF16=ON`); the default here is `Precision_High`, which is always safe.
    void setPrecision(BackendConfig::PrecisionMode precision);
    // Complete DiT evaluations, module rebuilds, and the milliseconds spent in each, for the last denoise run.
    void reportAccounting() const;

private:
    struct Resources;

    bool parseManifest();
    bool buildResourceTensors();
    std::shared_ptr<Module> loadModule(const std::string& name);
    // One denoising step: the packed forward over every partition, then the two Euler updates.
    bool denoiseStep(int step, VARP* videoRows, VARP* audioRows);
    VARP packedForward(VARP videoRows, VARP audioRows, VARP* audioVelocity, int step);
    // Euler update. The velocity is data-ward, so `x0 = x_t + (1 - t) * v` -- the opposite of the usual
    // flow-match sign. `skipRows` leading rows are copied through: they are conditioning anchors, and never
    // writing them is how the anchors survive the loop.
    VARP eulerStep(VARP sample, VARP velocity, int step, bool audio, int skipRows);
    bool writeLatents(VARP videoRows, VARP audioRows, const std::string& outputDir);

private:
    std::unique_ptr<Resources> mResources;
    std::shared_ptr<Module> mEmbed;
    std::vector<std::shared_ptr<Module>> mBlockGroups;
    std::shared_ptr<Module> mHead;
    Module::Config mModuleConfig;
    int mResidentGroups = 0;

    // Where a step's time goes. A windowed run rebuilds every partition once per step, so the split between
    // module construction and the forward itself is the difference between a kernel problem and a scheduling
    // one. `mForwardCount` is the number of complete DiT evaluations, which is the figure to compare against
    // another implementation's step count -- an audio microstep still runs the whole joint transformer.
    struct Accounting {
        int forwardCount = 0;
        int moduleLoads = 0;
        double loadMs = 0.0;
        double forwardMs = 0.0;
        void reset() { *this = Accounting(); }
    };
    Accounting mAccounting;
    BackendConfig::PrecisionMode mPrecision = BackendConfig::Precision_High;
    VARP mRopeCos;
    VARP mRopeSin;
    VARP mMask;
    VARP mTextMask;
    VARP mPromptEmbeds;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_MINIMAX_H3_DIFFUSION_HPP
