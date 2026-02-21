//  flux2_klein_diffusion.cpp - FLUX.2-Klein-4B diffusion model
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "diffusion/flux2_klein_diffusion.hpp"
#include "diffusion/diffusion_config.hpp"
#include "scheduler.hpp"
#include <rapidjson/document.h>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace CV;

namespace MNN {
namespace DIFFUSION {

// ===== Constructor =====
Flux2KleinDiffusion::Flux2KleinDiffusion(
    std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode,
    int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU,
    DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode,
    DiffusionCFGMode cfgMode, int numThreads)
    : Diffusion(modelPath, modelType, backendType, memoryMode,
                imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU,
                gpuMemoryMode, precisionMode, cfgMode, numThreads) {

    // Load base scheduler config (num_train_timesteps, shift, use_dynamic_shifting)
    loadSchedulerConfig();
    // Load Flux2Klein-specific scheduler fields (base_shift, max_shift, seq_len bounds)
    for (const auto& schedPath : {mModelPath + "/scheduler/scheduler_config.json",
                                   mModelPath + "/scheduler_config.json"}) {
        std::ifstream f(schedPath.c_str());
        if (!f.good()) continue;
        std::ostringstream oss; oss << f.rdbuf();
        rapidjson::Document doc; doc.Parse(oss.str().c_str());
        if (doc.HasParseError() || !doc.IsObject()) continue;
        if (doc.HasMember("base_shift") && (doc["base_shift"].IsFloat() || doc["base_shift"].IsDouble()))
            mBaseShift = (float)doc["base_shift"].GetDouble();
        if (doc.HasMember("max_shift") && (doc["max_shift"].IsFloat() || doc["max_shift"].IsDouble()))
            mMaxShift = (float)doc["max_shift"].GetDouble();
        if (doc.HasMember("base_image_seq_len") && doc["base_image_seq_len"].IsInt())
            mBaseImageSeqLen = doc["base_image_seq_len"].GetInt();
        if (doc.HasMember("max_image_seq_len") && doc["max_image_seq_len"].IsInt())
            mMaxImageSeqLen = doc["max_image_seq_len"].GetInt();
        break;
    }

    // Load VAE BN params from config.json vae.bn_mean / vae.bn_std
    {
        std::ifstream f((mModelPath + "/config.json").c_str());
        if (f.good()) {
            std::ostringstream oss; oss << f.rdbuf();
            rapidjson::Document doc; doc.Parse(oss.str().c_str());
            if (!doc.HasParseError() && doc.IsObject() &&
                doc.HasMember("vae") && doc["vae"].IsObject()) {
                auto& vae = doc["vae"];
                if (vae.HasMember("bn_mean") && vae["bn_mean"].IsArray()) {
                    for (auto& v : vae["bn_mean"].GetArray())
                        mVaeBnMean.push_back((float)v.GetDouble());
                }
                if (vae.HasMember("bn_std") && vae["bn_std"].IsArray()) {
                    for (auto& v : vae["bn_std"].GetArray())
                        mVaeBnStd.push_back((float)v.GetDouble());
                }
            }
        }
        if (!mVaeBnMean.empty())
            MNN_PRINT("[Flux2Klein] Loaded VAE BN params: %d channels\n", (int)mVaeBnMean.size());
        else
            MNN_PRINT("[Flux2Klein] Warning: VAE BN params not found in config.json\n");
    }

    // Image/latent dimensions.
    // Must align to 16: VAE scale=8 then patchify /2, so image must be multiple of 16
    // to ensure latent H/W are even integers.
    if (imageWidth > 0 && imageHeight > 0) {
        mImageWidth  = ((imageWidth  + 15) / 16) * 16;
        mImageHeight = ((imageHeight + 15) / 16) * 16;
        if (mImageWidth  < 256) mImageWidth  = 256;
        if (mImageHeight < 256) mImageHeight = 256;
    } else {
        mImageWidth = 512; mImageHeight = 512;
    }
    mLatentH = mImageHeight / 8;
    mLatentW = mImageWidth  / 8;
    MNN_PRINT("[Flux2Klein] image=%dx%d latent=%dx%d packed_seq=%d\n",
              mImageWidth, mImageHeight, mLatentH, mLatentW, (mLatentH/2)*(mLatentW/2));
}

Flux2KleinDiffusion::~Flux2KleinDiffusion() {
    // mTokenizer is unique_ptr, auto-destroyed
}

// ===== Scheduler helpers =====
float Flux2KleinDiffusion::computeEmpiricalMu(int imageSeqLen, int numSteps) const {
    float m = (mMaxShift - mBaseShift) / (float)(mMaxImageSeqLen - mBaseImageSeqLen);
    float b = mBaseShift - m * (float)mBaseImageSeqLen;
    return m * (float)imageSeqLen + b;
}

std::vector<float> Flux2KleinDiffusion::getSigmas(int numSteps, int imageSeqLen) const {
    std::vector<float> sigmas(numSteps + 1);
    for (int i = 0; i < numSteps; ++i) {
        float frac = (numSteps == 1) ? 0.f : (float)i / (float)(numSteps - 1);
        sigmas[i] = 1.0f - frac * (1.0f - 1.0f / (float)numSteps);
    }
    sigmas[numSteps] = 0.0f;
    if (mUseDynamicShifting) {
        float mu = computeEmpiricalMu(imageSeqLen, numSteps);
        float expMu = std::exp(mu);
        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            sigmas[i] = expMu * s / (expMu * s + (1.0f - s));
        }
    } else if (mFlowShift != 1.0f) {
        for (int i = 0; i < numSteps; ++i) {
            float s = sigmas[i];
            sigmas[i] = mFlowShift * s / (1.0f + (mFlowShift - 1.0f) * s);
        }
    }
    return sigmas;
}

// ===== Latent packing/unpacking =====
void Flux2KleinDiffusion::packLatentsToSeq(const float* src, float* dst,
                                            int B, int C, int H, int W) const {
    // [B,C,H,W] -> [B, H/2*W/2, C*4]
    int pH = H/2, pW = W/2, pC = C*4;
    for (int b = 0; b < B; ++b)
        for (int ph = 0; ph < pH; ++ph)
            for (int pw = 0; pw < pW; ++pw) {
                int si = ph*pW + pw;
                for (int c = 0; c < C; ++c)
                    for (int dh = 0; dh < 2; ++dh)
                        for (int dw = 0; dw < 2; ++dw)
                            dst[b*pH*pW*pC + si*pC + c*4+dh*2+dw] =
                                src[b*C*H*W + c*H*W + (ph*2+dh)*W + (pw*2+dw)];
            }
}

void Flux2KleinDiffusion::unpackLatentsToPatchified(const float* src, float* dst,
                                                     int B, int C, int H, int W, int seqLen) const {
    // [B, seqLen, C*4] -> [B, C*4, H/2, W/2]
    int pH = H/2, pW = W/2, pC = C*4;
    for (int b = 0; b < B; ++b)
        for (int ph = 0; ph < pH; ++ph)
            for (int pw = 0; pw < pW; ++pw) {
                int si = ph*pW + pw;
                for (int fc = 0; fc < pC; ++fc)
                    dst[b*pC*pH*pW + fc*pH*pW + ph*pW+pw] =
                        src[b*seqLen*pC + si*pC + fc];
            }
}

void Flux2KleinDiffusion::prepareImgIds(float* dst, int H, int W, int seqOffset, float t_coord) const {
    // img_ids format: (t, h, w, l=0)
    // For noise latents: t=0 (_prepare_latent_ids)
    // For image latents: t=IMAGE_LATENT_T_OFFSET=10 (_prepare_image_ids, scale=10, first image)
    int pH = H/2, pW = W/2;
    for (int h = 0; h < pH; ++h)
        for (int w = 0; w < pW; ++w) {
            int idx = seqOffset + h*pW + w;
            dst[idx*ID_DIM+0]=t_coord;   // t
            dst[idx*ID_DIM+1]=(float)h;  // h
            dst[idx*ID_DIM+2]=(float)w;  // w
            dst[idx*ID_DIM+3]=0.f;       // l=0
        }
}

void Flux2KleinDiffusion::prepareTxtIds(float* dst, int seqLen) const {
    // txt_ids format: (t=0, h=0, w=0, l=i) matching pipeline _prepare_text_ids
    for (int i = 0; i < seqLen; ++i) {
        dst[i*ID_DIM+0]=0.f;       // t=0
        dst[i*ID_DIM+1]=0.f;       // h=0
        dst[i*ID_DIM+2]=0.f;       // w=0
        dst[i*ID_DIM+3]=(float)i;  // l=i
    }
}


// ===== load() =====
bool Flux2KleinDiffusion::load() {
    AUTOTIME;
    // Enable flash attention (ATTENTION_OPTION=8) to avoid O(N^2) memory for large seqLen
    // (e.g. 8192 for 1024 edit). Flash attention uses block size 64.
    if (!initRuntimeManagers(/*gpuBufferMode=*/true, /*attentionHint=*/8)) return false;

    Module::Config mc; mc.shapeMutable = true;
    DiffusionConfig diff_config(mModelPath);
    mModules.resize(4);

    // [0] Text encoder - input: input_ids[B,S] + attention_mask[B,S], output: prompt_embeds[B,S,H]
    {
        auto path = diff_config.text_encoder_model();
        MNN_PRINT("[Flux2Klein] Load text encoder: %s\n", path.c_str());
        Module::Config tec; tec.shapeMutable = true;
        // Use CPU runtime for text encoder if requested
        auto& te_runtime = runtime_manager_cpu_ ? runtime_manager_cpu_ : runtime_manager_;
        mModules[0].reset(Module::load({"input_ids","attention_mask"},{"prompt_embeds"}, path.c_str(), te_runtime, &tec));
        if (!mModules[0]) { MNN_ERROR("[Flux2Klein] Failed to load text encoder\n"); return false; }
    }

    // [1] Transformer - dynamic shape, inputs: hidden_states/timestep/encoder_hidden_states/txt_ids/img_ids
    {
        auto path = diff_config.unet_model();
        MNN_PRINT("[Flux2Klein] Load transformer: %s\n", path.c_str());
        mModules[1].reset(Module::load(
            {"hidden_states","timestep","encoder_hidden_states","txt_ids","img_ids"},
            {"output"}, path.c_str(), runtime_manager_, &mc));  // shared_ptr overload
        if (!mModules[1]) { MNN_ERROR("[Flux2Klein] Failed to load transformer\n"); return false; }
    }
    // [2] VAE decoder - input: [B,128,H/2,W/2] normalized patchified (BN denorm+unpatchify baked in)
    {
        auto path = diff_config.vae_decoder_model();
        MNN_PRINT("[Flux2Klein] Load VAE decoder: %s\n", path.c_str());
        auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
        mModules[2].reset(Module::load({"latent_sample"},{"sample"}, path.c_str(), vae_runtime, &mc));
        if (!mModules[2]) { MNN_ERROR("[Flux2Klein] Failed to load VAE decoder\n"); return false; }
        mModules[2]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    // [3] VAE encoder - input: [B,3,H,W] normalized, output: [B,32,H/8,W/8]
    {
        auto path = diff_config.vae_encoder_model();
        MNN_PRINT("[Flux2Klein] Load VAE encoder: %s\n", path.c_str());
        auto& vae_runtime = runtime_manager_vae_cpu_ ? runtime_manager_vae_cpu_ : runtime_manager_;
        mModules[3].reset(Module::load({"sample"},{"latent_sample"}, path.c_str(), vae_runtime, &mc));
        if (!mModules[3]) { MNN_ERROR("[Flux2Klein] Failed to load VAE encoder\n"); return false; }
        mModules[3]->traceOrOptimize(MNN::Interpreter::Session_Resize_Fix);
    }
    return true;
}

// ===== Text Encoder =====
// Hardcoded Qwen3 chat_template (user-only, no thinking, no system prompt):
//   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
// This matches: tokenizer.apply_chat_template([{"role":"user","content":prompt}],
//   tokenize=False, add_generation_prompt=True, enable_thinking=False)
VARP Flux2KleinDiffusion::text_encoder_llm(const std::string& prompt) {
    AUTOTIME;
    if (!mModules[0]) { MNN_PRINT("[Flux2Klein] Error: text encoder not loaded\n"); return nullptr; }

    std::vector<int> inputIds;

#ifdef MNN_BUILD_LLM
    if (!mTokenizer) {
        std::string tokPath = mModelPath + "/tokenizer.txt";
        mTokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokPath));
        if (mTokenizer)
            MNN_PRINT("[Flux2Klein] Tokenizer loaded: %s\n", tokPath.c_str());
        else
            MNN_PRINT("[Flux2Klein] Warning: tokenizer load failed: %s\n", tokPath.c_str());
    }
    if (mTokenizer) {
        // Apply Qwen3 chat_template (hardcoded, user-only, no thinking)
        std::string templated = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        inputIds = mTokenizer->encode(templated);
        MNN_PRINT("[Flux2Klein] Tokens after chat_template: %d\n", (int)inputIds.size());
    }
#endif

    if (inputIds.empty()) {
        MNN_PRINT("[Flux2Klein] Warning: tokenize failed, using dummy tokens\n");
        inputIds.resize(mTextSeqLen, 0);
    }

    int actualLen = (int)inputIds.size();
    // Truncate if needed
    if (actualLen > mTextSeqLen) {
        inputIds.resize(mTextSeqLen);
        actualLen = mTextSeqLen;
    }
    // Pad to mTextSeqLen (matching Python: padding='max_length', max_length=512)
    // padding token id = 0, attention_mask = 0 for padding positions
    int seqLen = mTextSeqLen;
    inputIds.resize(seqLen, 0);

    // Build input_ids [1, seqLen] and attention_mask [1, seqLen]
    INTS idShape = {1, seqLen};
    auto idsVar  = _Input(idShape, NCHW, halide_type_of<int>());
    auto maskVar = _Input(idShape, NCHW, halide_type_of<int>());
    auto idsPtr  = idsVar->writeMap<int>();
    auto maskPtr = maskVar->writeMap<int>();
    for (int i = 0; i < seqLen; ++i) {
        idsPtr[i]  = inputIds[i];
        maskPtr[i] = (i < actualLen) ? 1 : 0;
    }

    // Forward through text encoder module
    auto outs = mModules[0]->onForward({idsVar, maskVar});
    if (outs.empty()) { MNN_PRINT("[Flux2Klein] Text encoder forward failed\n"); return nullptr; }

    // Output: prompt_embeds [1, seqLen, hiddenSize]
    auto embeds = _Convert(outs[0], NCHW);
    embeds.fix(VARP::CONSTANT);
    auto fi = embeds->getInfo();
    MNN_PRINT("[Flux2Klein] Text embeds: [%d,%d,%d]\n", fi->dim[0], fi->dim[1], fi->dim[2]);
    return embeds;
}

// ===== VAE Decoder =====
VARP Flux2KleinDiffusion::vae_decoder(VARP latent) {
    if (mMemoryMode != 1) mModules[1].reset();
    AUTOTIME;
    auto outs = mModules[2]->onForward({latent});
    if (outs.empty()) { MNN_PRINT("[Flux2Klein] VAE decode failed\n"); return nullptr; }
    return nchwFloatToHwcBGR(_Convert(outs[0], NCHW));
}

// ===== VAE Encoder =====
// Returns normalized patchified latent [1, 128, H/2, W/2] matching Python _encode_vae_image:
//   1. vae.encode(image) -> [1, 32, H/8, W/8]
//   2. _patchify_latents -> [1, 128, H/16, W/16]  (H/8 / 2 = H/16)
//   3. BN normalize: (x - bn_mean) / bn_std
VARP Flux2KleinDiffusion::vae_encoder(VARP image) {
    AUTOTIME;
    auto outs = mModules[3]->onForward({image});
    if (outs.empty()) { MNN_PRINT("[Flux2Klein] VAE encode failed\n"); return nullptr; }
    auto lat = _Convert(outs[0], NCHW);  // [1, 32, H/8, W/8]
    lat.fix(VARP::CONSTANT);

    // Patchify: [1,32,H,W] -> [1,128,H/2,W/2]
    auto info = lat->getInfo();
    int B = info->dim[0], C = info->dim[1], H = info->dim[2], W = info->dim[3];
    int pH = H/2, pW = W/2, pC = C*4;
    std::vector<float> patchData(B * pC * pH * pW);
    const float* src = lat->readMap<float>();
    // _patchify_latents: view(B,C,H/2,2,W/2,2).permute(0,1,3,5,2,4).reshape(B,C*4,H/2,W/2)
    // Result channel index: c*4 + dh*2 + dw
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            for (int ph = 0; ph < pH; ++ph)
                for (int pw = 0; pw < pW; ++pw)
                    for (int dh = 0; dh < 2; ++dh)
                        for (int dw = 0; dw < 2; ++dw) {
                            int srcIdx = b*C*H*W + c*H*W + (ph*2+dh)*W + (pw*2+dw);
                            int fc = c*4 + dh*2 + dw;
                            int dstIdx = b*pC*pH*pW + fc*pH*pW + ph*pW + pw;
                            patchData[dstIdx] = src[srcIdx];
                        }

    // BN normalize: (x - bn_mean) / bn_std  (per-channel, broadcast over H,W)
    if ((int)mVaeBnMean.size() == pC && (int)mVaeBnStd.size() == pC) {
        for (int b = 0; b < B; ++b)
            for (int c = 0; c < pC; ++c) {
                float mean = mVaeBnMean[c], bnStd = mVaeBnStd[c];
                for (int i = 0; i < pH*pW; ++i)
                    patchData[b*pC*pH*pW + c*pH*pW + i] = (patchData[b*pC*pH*pW + c*pH*pW + i] - mean) / bnStd;
            }
    } else {
        MNN_PRINT("[Flux2Klein] Warning: BN params missing, skipping BN normalize\n");
    }

    auto patchVar = _Input({B, pC, pH, pW}, NCHW, halide_type_of<float>());
    memcpy(patchVar->writeMap<float>(), patchData.data(), patchData.size()*sizeof(float));
    patchVar.fix(VARP::CONSTANT);
    MNN_PRINT("[Flux2Klein] VAE encoded+patchified: [%d,%d,%d,%d]\n", B, pC, pH, pW);
    return patchVar;
}


// ===== UNet (Denoising Loop) =====
VARP Flux2KleinDiffusion::unet(VARP textEmbeds, VARP imageLatents,
                                int iterNum, int randomSeed,
                                std::function<void(int)> progressCallback) {
    // Free tokenizer to save memory before denoising
#ifdef MNN_BUILD_LLM
    if (mMemoryMode != 1 && mTokenizer) {
        mTokenizer.reset();
        MNN_PRINT("[Flux2Klein] Tokenizer unloaded\n");
    }
#endif
    // Copy textEmbeds to independent CPU tensor before freeing text encoder.
    // textEmbeds VARP may reference memory owned by mModules[0]; reset() frees it.
    if (mMemoryMode != 1 && textEmbeds.get()) {
        auto info = textEmbeds->getInfo();
        if (info) {
            size_t n = 1; for (auto d : info->dim) n *= d;
            std::vector<float> buf(n);
            const float* src = textEmbeds->readMap<float>();
            if (src) memcpy(buf.data(), src, n * sizeof(float));
            VARP tmp = _Input(info->dim, info->order, halide_type_of<float>());
            memcpy(tmp->writeMap<float>(), buf.data(), n * sizeof(float));
            tmp.fix(VARP::CONSTANT);
            textEmbeds = tmp;
        }
        mModules[0].reset();
    }
    bool isT2I = !imageLatents.get();
    int pH = mLatentH/2, pW = mLatentW/2;
    int singleSeq = pH * pW;
    int imgSeqLen = isT2I ? singleSeq : singleSeq * 2;
    int pC = mPackedC;  // 128

    MNN_PRINT("[Flux2Klein] %s: singleSeq=%d imgSeqLen=%d textSeqLen=%d\n",
              isT2I ? "T2I" : "Edit", singleSeq, imgSeqLen, mTextSeqLen);

    // Generate noise [B,C,H,W] with PhiloxRNG (aligned with PyTorch)
    int latentSize = mLatentC * mLatentH * mLatentW;
    std::vector<float> noiseData(latentSize);
    {
        int seed = randomSeed < 0 ? std::random_device()() : randomSeed;
        generateLatentNoise(noiseData.data(), latentSize, seed);
    }

    // Compute sigmas
    auto sigmas = getSigmas(iterNum, singleSeq);

    // Prepare txt_ids [1, textSeqLen, 4] - batch dim included
    // Use actual text embed seq len (not mTextSeqLen)
    int txtSeqLen = textEmbeds->getInfo()->dim[1];
    std::vector<float> txtIdsData(txtSeqLen * ID_DIM, 0.f);
    prepareTxtIds(txtIdsData.data(), txtSeqLen);

    // Prepare img_ids [1, imgSeqLen, 4] - batch dim included
    std::vector<float> imgIdsData(imgSeqLen * ID_DIM, 0.f);
    // Noise latent ids: t=0 (matches _prepare_latent_ids)
    prepareImgIds(imgIdsData.data(), mLatentH, mLatentW, 0, 0.f);
    if (!isT2I) {
        // Image latent ids: t=IMAGE_LATENT_T_OFFSET=10 (matches _prepare_image_ids, scale=10, first image)
        prepareImgIds(imgIdsData.data(), mLatentH, mLatentW, singleSeq, (float)Flux2KleinDiffusion::IMAGE_LATENT_T_OFFSET);
    }

    // Pre-pack image latents for editing
    // vae_encoder returns patchified+BN-normalized [1, 128, pH, pW] (pH=H/16, pW=W/16)
    // _pack_latents: [B, C, H, W] -> reshape(B, C, H*W).permute(0,2,1) -> [B, H*W, C]
    std::vector<float> imageLatentsPacked;
    if (!isT2I) {
        auto imgInfo = imageLatents->getInfo();
        int iB = imgInfo->dim[0], iC = imgInfo->dim[1], iH = imgInfo->dim[2], iW = imgInfo->dim[3];
        int imgSeq = iH * iW;
        MNN_ASSERT(imgSeq == singleSeq && iC == pC);
        imageLatentsPacked.resize(singleSeq * pC);
        const float* imgSrc = imageLatents->readMap<float>();
        // [1, pC, pH, pW] -> [pH*pW, pC]  (permute C and HW)
        for (int c = 0; c < pC; ++c)
            for (int i = 0; i < singleSeq; ++i)
                imageLatentsPacked[i * pC + c] = imgSrc[c * singleSeq + i];
    }

    // Pack noise latents: [B,C,H,W] -> [singleSeq, pC]
    std::vector<float> latentSeq(singleSeq * pC);
    packLatentsToSeq(noiseData.data(), latentSeq.data(), 1, mLatentC, mLatentH, mLatentW);

    // Allocate transformer inputs - ids shape [1, seq, 4] with batch dim
    INTS txtIdsShape = {1, txtSeqLen, ID_DIM};
    auto txtIdsVar = _Input(txtIdsShape, NCHW, halide_type_of<float>());
    memcpy(txtIdsVar->writeMap<float>(), txtIdsData.data(), txtIdsData.size()*sizeof(float));

    INTS imgIdsShape = {1, imgSeqLen, ID_DIM};
    auto imgIdsVar = _Input(imgIdsShape, NCHW, halide_type_of<float>());
    memcpy(imgIdsVar->writeMap<float>(), imgIdsData.data(), imgIdsData.size()*sizeof(float));

    // Current latent state [1, singleSeq, pC]
    std::vector<float> curLatent = latentSeq;

    MNN_PRINT("[Flux2Klein] Denoising %d steps, sigma[0]=%.4f sigma[-1]=%.4f\n",
              iterNum, sigmas[0], sigmas[iterNum-1]);

    for (int i = 0; i < iterNum; ++i) {
        AUTOTIME;
        float sigma      = sigmas[i];
        float sigma_next = sigmas[i + 1];
        float dt         = sigma_next - sigma;

        // Build sample input [1, imgSeqLen, pC]
        std::vector<float> sampleData(imgSeqLen * pC);
        memcpy(sampleData.data(), curLatent.data(), singleSeq * pC * sizeof(float));
        if (!isT2I) {
            memcpy(sampleData.data() + singleSeq * pC,
                   imageLatentsPacked.data(), singleSeq * pC * sizeof(float));
        }

        auto sampleVar   = _Input({1, imgSeqLen, pC}, NCHW, halide_type_of<float>());
        auto timestepVar = _Input({1}, NCHW, halide_type_of<float>());
        memcpy(sampleVar->writeMap<float>(), sampleData.data(), sampleData.size()*sizeof(float));
        timestepVar->writeMap<float>()[0] = sigma;  // transformer expects sigma in [0,1]

        std::vector<VARP> inputs = {sampleVar, timestepVar, textEmbeds, txtIdsVar, imgIdsVar};
        auto outs = mModules[1]->onForward(inputs);
        if (outs.empty()) {
            MNN_PRINT("[Flux2Klein] Transformer failed at step %d\n", i);
            return nullptr;
        }
        auto noisePred = _Convert(outs[0], NCHW);  // [1, imgSeqLen, pC]

        // For editing: take only first singleSeq tokens
        if (!isT2I) {
            std::vector<int> starts = {0, 0, 0};
            std::vector<int> sizes  = {1, singleSeq, pC};
            noisePred = _Slice(noisePred,
                _Const(starts.data(),{3},NCHW,halide_type_of<int>()),
                _Const(sizes.data(), {3},NCHW,halide_type_of<int>()));
        }
        noisePred.fix(VARP::CONSTANT);

        // Euler update: latent = latent + dt * noise_pred
        const float* npPtr = noisePred->readMap<float>();
        for (int j = 0; j < singleSeq * pC; ++j)
            curLatent[j] += dt * npPtr[j];

        if (mBackendType == MNN_FORWARD_OPENCL && (i+1) % 2 == 0)
            MNN::Express::ExecutorScope::Current()->gc(MNN::Express::Executor::PART);

        if (progressCallback) progressCallback((2 + i) * 100 / (iterNum + 3));
        MNN_PRINT("[Flux2Klein] Step %d/%d sigma=%.4f\n", i+1, iterNum, sigma);
    }

    // Unpack: [singleSeq, pC] -> [1, pC, pH, pW]  (normalized patchified for VAE)
    int pH2 = mLatentH/2, pW2 = mLatentW/2;
    std::vector<float> patchifiedData(pC * pH2 * pW2);
    unpackLatentsToPatchified(curLatent.data(), patchifiedData.data(),
                              1, mLatentC, mLatentH, mLatentW, singleSeq);

    auto patchifiedVar = _Input({1, pC, pH2, pW2}, NCHW, halide_type_of<float>());
    memcpy(patchifiedVar->writeMap<float>(), patchifiedData.data(), patchifiedData.size()*sizeof(float));
    patchifiedVar.fix(VARP::CONSTANT);
    return patchifiedVar;
}


// ===== run() overloads =====
bool Flux2KleinDiffusion::run(const std::string prompt, const std::string imagePath,
                               int iterNum, int randomSeed,
                               std::function<void(int)> progressCallback) {
    return run(prompt, imagePath, iterNum, randomSeed, 1.0f, progressCallback, "");
}

bool Flux2KleinDiffusion::run(const std::string prompt, const std::string outputPath,
                               int iterNum, int randomSeed, float cfgScale,
                               std::function<void(int)> progressCallback,
                               const std::string inputImagePath) {
    AUTOTIME;
    bool isT2I = inputImagePath.empty();
    VARP imageLatents = nullptr;

    if (!isT2I) {
        MNN_PRINT("[Flux2Klein] Image Edit: %s\n", inputImagePath.c_str());
        auto rawImage = CV::imread(inputImagePath);
        if (!rawImage.get()) {
            MNN_PRINT("[Flux2Klein] Error: cannot load %s\n", inputImagePath.c_str());
            return false;
        }
        auto processed = resizeAndCenterCrop(rawImage, mImageWidth, mImageHeight);
        auto rgbImage  = bgrToRgb(processed);
        auto inputNorm = hwcToNchw(rgbImage, true);  // normalize to [-1,1]
        imageLatents = vae_encoder(inputNorm);
        if (!imageLatents.get()) { MNN_PRINT("[Flux2Klein] VAE encode failed\n"); return false; }
        // Copy imageLatents to independent CPU tensor before freeing VAE encoder.
        // imageLatents VARP may reference memory owned by mModules[3]; reset() frees it.
        {
            auto info = imageLatents->getInfo();
            if (info) {
                size_t n = 1; for (auto d : info->dim) n *= d;
                std::vector<float> buf(n);
                const float* src = imageLatents->readMap<float>();
                if (src) memcpy(buf.data(), src, n * sizeof(float));
                VARP tmp = _Input(info->dim, info->order, halide_type_of<float>());
                memcpy(tmp->writeMap<float>(), buf.data(), n * sizeof(float));
                tmp.fix(VARP::CONSTANT);
                imageLatents = tmp;
            }
        }
        if (mMemoryMode != 1) mModules[3].reset();
    }

    if (iterNum < 1)  iterNum = 8;
    if (iterNum > 50) iterNum = 50;

    if (progressCallback) progressCallback(0);

    auto textEmbeds = text_encoder_llm(prompt);
    if (!textEmbeds.get()) { MNN_PRINT("[Flux2Klein] Text encode failed\n"); return false; }
    if (progressCallback) progressCallback(1 * 100 / (iterNum + 3));
    auto latent = unet(textEmbeds, imageLatents, iterNum, randomSeed, progressCallback);
    if (!latent.get()) { MNN_PRINT("[Flux2Klein] UNet failed\n"); return false; }

    auto image = vae_decoder(latent);
    if (!image.get()) { MNN_PRINT("[Flux2Klein] VAE decode failed\n"); return false; }

    bool res = CV::imwrite(outputPath, image);
    if (res) MNN_PRINT("[Flux2Klein] Saved: %s\n", outputPath.c_str());
    else     MNN_PRINT("[Flux2Klein] Error: imwrite failed: %s\n", outputPath.c_str());

    if (mMemoryMode != 1) mModules[2].reset();
    if (progressCallback) progressCallback(100);
    return res;
}

bool Flux2KleinDiffusion::run(const VARP input_embeds, const std::string& mode,
                               const std::string& inputImagePath, const std::string& outputImagePath,
                               int width, int height, int iterNum, int randomSeed,
                               bool use_cfg, float cfg_scale,
                               std::function<void(int)> progressCallback) {
    MNN_PRINT("[Flux2Klein] input_embeds interface not supported\n");
    return false;
}

} // namespace DIFFUSION
} // namespace MNN
