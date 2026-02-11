//
//  longcat_diffusion.hpp
//  LongCat Image Edit / T2I diffusion model (Flux-like architecture)
//
//  Created for OfflineAI project
//
#ifndef MNN_LONGCAT_DIFFUSION_HPP
#define MNN_LONGCAT_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>

namespace MNN {
namespace DIFFUSION {

class Tokenizer;

class MNN_PUBLIC LongCatDiffusion : public Diffusion {
public:
    LongCatDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
                     int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
                     DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
                     DiffusionCFGMode cfgMode, int numThreads = 4);
    virtual ~LongCatDiffusion();
    
    virtual bool load() override;
    virtual bool run(const std::string prompt, const std::string imagePath, int iterNum, int randomSeed, std::function<void(int)> progressCallback) override;
    virtual bool run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath = "") override;
    virtual bool run(const VARP input_embeds, const std::string& mode, const std::string& inputImagePath,
                    const std::string& outputImagePath, int width, int height, int iterNum, int randomSeed,
                    bool use_cfg, float cfg_scale, std::function<void(int)> progressCallback) override;

private:
    VARP text_encoder_llm(const std::string& prompt, VARP preprocessedImage);
    VARP unet(VARP text_embeddings, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
    VARP applyEulerUpdate(VARP sample, VARP noise_pred, float dt);
    VARP unpackLatentsGPU(VARP packed, int B, int C, int H, int W);
    void getCFGSigmaRange(float& sigmaLow, float& sigmaHigh) const;

private:
    int mMaxTextLen = 128;
    int mTrainTimestepsNum = 1000;
    float mFlowShift = 3.0f;
    bool mUseDynamicShifting = false;
    int mLatentC = 16;
    int mLatentH = 128;
    int mLatentW = 128;
    float mDefaultCfgScale = 4.5f;
    
    std::vector<float> mSigmas;
    std::vector<int> mTimeSteps;
    VARP mLatentVar, mPromptVar, mAttentionMaskVar, mTimestepVar, mSampleVar;
    VARP mTxtIdsVar, mImgIdsVar;
    VARP mImageLatentsVar;
    std::vector<float> mInitNoise;
    
    UNetPreprocessFunc mUNetPreprocess;
    SchedulerType mSchedulerType = SCHEDULER_EULER;
    
    // LongCat config
    std::string mTextEncoderDir = "text_encoder";
    LLMEncoderConfig mLlmEncoderConfig;
    void* mLlm = nullptr;  // Llm* pointer, void* to avoid header dependency
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_LONGCAT_DIFFUSION_HPP
