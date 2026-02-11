//  flux2_klein_diffusion.hpp - FLUX.2-Klein-4B diffusion model
#ifndef MNN_FLUX2_KLEIN_DIFFUSION_HPP
#define MNN_FLUX2_KLEIN_DIFFUSION_HPP

#include "diffusion.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#ifdef MNN_BUILD_LLM
#include "llm/tokenizer.hpp"
#endif

namespace MNN {
namespace DIFFUSION {

class MNN_PUBLIC Flux2KleinDiffusion : public Diffusion {
public:
    Flux2KleinDiffusion(std::string modelPath, DiffusionModelType modelType,
                        MNNForwardType backendType, int memoryMode,
                        int imageWidth, int imageHeight,
                        bool textEncoderOnCPU, bool vaeOnCPU,
                        DiffusionGpuMemoryMode gpuMemoryMode,
                        DiffusionPrecisionMode precisionMode,
                        DiffusionCFGMode cfgMode, int numThreads = 4);
    virtual ~Flux2KleinDiffusion();

    virtual bool load() override;

    virtual bool run(const std::string prompt, const std::string imagePath,
                     int iterNum, int randomSeed,
                     std::function<void(int)> progressCallback) override;

    virtual bool run(const std::string prompt, const std::string outputPath,
                     int iterNum, int randomSeed, float cfgScale,
                     std::function<void(int)> progressCallback,
                     const std::string inputImagePath = "") override;

    virtual bool run(const VARP input_embeds, const std::string& mode,
                     const std::string& inputImagePath, const std::string& outputImagePath,
                     int width, int height, int iterNum, int randomSeed,
                     bool use_cfg, float cfg_scale,
                     std::function<void(int)> progressCallback) override;

private:
    VARP text_encoder_llm(const std::string& prompt);
    VARP unet(VARP textEmbeds, VARP imageLatents,
              int iterNum, int randomSeed,
              std::function<void(int)> progressCallback);
    VARP vae_decoder(VARP latent);
    VARP vae_encoder(VARP image);
    float computeEmpiricalMu(int imageSeqLen, int numSteps) const;
    std::vector<float> getSigmas(int numSteps, int imageSeqLen) const;

    void packLatentsToSeq(const float* src, float* dst, int B, int C, int H, int W) const;
    void unpackLatentsToPatchified(const float* src, float* dst,
                                   int B, int C, int H, int W, int seqLen) const;
    void prepareImgIds(float* dst, int H, int W, int seqOffset = 0, float t_coord = 0.f) const;
    void prepareTxtIds(float* dst, int seqLen) const;

    // Flux2Klein-specific scheduler params (base params are in Diffusion base class)
    float mBaseShift          = 0.5f;
    float mMaxShift           = 1.15f;
    int   mBaseImageSeqLen    = 256;
    int   mMaxImageSeqLen     = 4096;

    // Latent dims
    int mLatentC = 32;   // VAE latent channels (before patchify)
    int mPackedC = 128;  // C*4 after patchify
    int mLatentH = 64;   // H/8
    int mLatentW = 64;   // W/8

    // Text encoder
    // chat_template hardcoded: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    int mTextSeqLen = 512;
#ifdef MNN_BUILD_LLM
    std::unique_ptr<MNN::Transformer::Tokenizer> mTokenizer;
#endif

    // VAE BN normalization params (patchify + BN normalize after VAE encode)
    // Loaded from config.json vae.bn_mean / vae.bn_std (128 channels)
    std::vector<float> mVaeBnMean;
    std::vector<float> mVaeBnStd;

    static const int ID_DIM = 4;              // FLUX.2-Klein uses 4D ids
    static const int IMAGE_LATENT_T_OFFSET = 10; // _prepare_image_ids: t=scale+scale*i, scale=10
};

} // namespace DIFFUSION
} // namespace MNN

#endif // MNN_FLUX2_KLEIN_DIFFUSION_HPP
