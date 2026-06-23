//
//  wan_diffusion.hpp
//
//  Wan2.1 text-to-video runtime skeleton.
//
#ifndef MNN_WAN_DIFFUSION_HPP
#define MNN_WAN_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>

namespace MNN {
namespace DIFFUSION {

class Tokenizer;

class MNN_PUBLIC WanDiffusion : public Diffusion {
public:
    WanDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual ~WanDiffusion();

    virtual bool load() override;

    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback) override;

    virtual bool run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                     const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                     bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) override;

    bool runVideo(const std::string& prompt, const std::string& outputDir, int width, int height, int frames, int steps,
                  int seed, float cfgScale, std::function<void(int)> progressCallback) override;

private:
    VARP encodePrompt(const std::string& prompt, int* seqLen, VARP* outMask);
    VARP transformer(VARP hiddenStates, VARP timestep, VARP encoderHiddenStates, VARP encoderAttentionMask);
    VARP vaeDecoder(VARP latent);
    VARP stepFlowMatch(VARP sample, VARP modelOutput, int index);
    bool saveVideoFrames(VARP sample, const std::string& outputDir, int requestedFrames);
    bool saveFrame(VARP image4d, const std::string& fileName);

private:
    std::unique_ptr<Tokenizer> mTokenizer;
    int mMaxTextLen = 512;
    // Latent channel count for the Wan VAE/transformer pair. 16 matches the
    // Wan2.1-T2V-1.3B variant used today; for other Wan variants (e.g. 5B)
    // this should be sourced from model metadata or WanComponents.
    // TODO: read from a config file once metadata plumbing lands.
    int mLatentChannels = 16;
    std::vector<float> mTimesteps;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_WAN_DIFFUSION_HPP
