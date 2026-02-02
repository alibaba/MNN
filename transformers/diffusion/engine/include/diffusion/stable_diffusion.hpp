//
//  stable_diffusion.hpp
//
//  Created by MNN on 2025/01/12.
//  MNN
//
#ifndef MNN_STABLE_DIFFUSION_HPP
#define MNN_STABLE_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>

namespace MNN {
namespace DIFFUSION {

class Tokenizer;
class Scheduler;

class MNN_PUBLIC StableDiffusion : public Diffusion {
public:
    StableDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode);
    virtual ~StableDiffusion() = default;
    
    virtual bool load() override;

    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) override;
    
    // 统一的生成接口
    virtual bool run(const VARP input_embeds, 
                    const std::string& mode,
                    const std::string& inputImagePath,
                    const std::string& outputImagePath,
                    int width,
                    int height,
                    int iterNum,
                    int randomSeed,
                    bool use_cfg,
                    float cfg_scale,
                    std::function<void(int)> progressCallback) override;

private:
    VARP step_plms(VARP sample, VARP model_output, int index);
    VARP text_encoder(const std::vector<int>& ids);
    VARP unet(VARP text_embeddings, int iterNum, int randomSeed, std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);

private:
    std::vector<int> mTimeSteps;
    std::vector<float> mAlphas;
    std::vector<VARP> mEts;
    VARP mSample;
    VARP mLatentVar, mPromptVar, mTimestepVar, mSampleVar;
    std::vector<float> mInitNoise;
    int mMaxTextLen = 77;
    std::unique_ptr<Tokenizer> mTokenizer;
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_STABLE_DIFFUSION_HPP
