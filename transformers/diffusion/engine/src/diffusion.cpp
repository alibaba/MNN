#include <random>
#include <fstream>
#include <chrono>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "diffusion/diffusion.hpp"
#include "diffusion/stable_diffusion.hpp"
#include "diffusion/sana_diffusion.hpp"

// Forward declarations for our custom subclasses
namespace MNN { namespace DIFFUSION {
class ZImageDiffusion;
class LongCatDiffusion;
}}

#include "diffusion/zimage_diffusion.hpp"
#include "diffusion/longcat_diffusion.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <cv/cv.hpp>

#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#endif

using namespace CV;

namespace MNN {
namespace DIFFUSION {

// Helper function for progress display
void display_progress(int cur, int total){
    putchar('\r');
    MNN_PRINT("[");
    for (int i = 0; i < cur; i++) putchar('#');
    for (int i = 0; i < total - cur; i++) putchar('-');
    MNN_PRINT("]");
    fprintf(stdout, "  [%3d%%]", cur * 100 / total);
    if (cur == total) putchar('\n');
    fflush(stdout);
}

// PhiloxRNG is now defined in diffusion.hpp header

// ===== Base Diffusion Implementation =====

Diffusion::Diffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) :
mModelPath(modelPath), mModelType(modelType), mBackendType(backendType), mMemoryMode(memoryMode) {
}

Diffusion::~Diffusion() {
    mModules.clear();
    runtime_manager_.reset();
    runtime_manager_cpu_.reset();
}

// Default extended run: delegates to simple run (subclasses can override)
bool Diffusion::run(const std::string prompt, const std::string outputPath, int iterNum, int randomSeed, float cfgScale, std::function<void(int)> progressCallback, const std::string inputImagePath) {
    // Default: ignore cfgScale and inputImagePath, delegate to simple run
    return run(prompt, outputPath, iterNum, randomSeed, progressCallback);
}

// ===== Factory Methods =====

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode) {
    if (modelType == SANA_DIFFUSION) {
        return new SanaDiffusion(modelPath, modelType, backendType, memoryMode);
    } else if (modelType == STABLE_DIFFUSION_ZIMAGE) {
        return new ZImageDiffusion(modelPath, modelType, backendType, memoryMode, 0, 0, true, false, GPU_MEMORY_AUTO, PRECISION_AUTO, CFG_MODE_AUTO, 4);
    } else if (modelType == LONGCAT_IMAGE_EDIT) {
        return new LongCatDiffusion(modelPath, modelType, backendType, memoryMode, 0, 0, true, false, GPU_MEMORY_AUTO, PRECISION_AUTO, CFG_MODE_AUTO, 4);
    } else {
        return new StableDiffusion(modelPath, modelType, backendType, memoryMode);
    }
}

Diffusion* Diffusion::createDiffusion(std::string modelPath, DiffusionModelType modelType, MNNForwardType backendType, int memoryMode, int imageWidth, int imageHeight, bool textEncoderOnCPU, bool vaeOnCPU, DiffusionGpuMemoryMode gpuMemoryMode, DiffusionPrecisionMode precisionMode, DiffusionCFGMode cfgMode, int numThreads) {
    if (modelType == STABLE_DIFFUSION_ZIMAGE) {
        return new ZImageDiffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU, gpuMemoryMode, precisionMode, cfgMode, numThreads);
    } else if (modelType == LONGCAT_IMAGE_EDIT) {
        return new LongCatDiffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU, gpuMemoryMode, precisionMode, cfgMode, numThreads);
    } else if (modelType == SANA_DIFFUSION) {
        return new SanaDiffusion(modelPath, modelType, backendType, memoryMode, imageWidth, imageHeight, textEncoderOnCPU, vaeOnCPU, gpuMemoryMode, precisionMode, cfgMode, numThreads);
    } else {
        return new StableDiffusion(modelPath, modelType, backendType, memoryMode);
    }
}

// ===== Image Processing Utility Functions =====

VARP Diffusion::resizeAndCenterCrop(VARP image, int targetW, int targetH) {
    auto info = image->getInfo();
    if (!info || info->dim.size() != 3) {
        MNN_ERROR("resizeAndCenterCrop: Invalid input shape\n");
        return nullptr;
    }
    int origH = info->dim[0], origW = info->dim[1], origC = info->dim[2];
    float aspectRatio = static_cast<float>(origW) / origH;
    float targetAspect = static_cast<float>(targetW) / targetH;
    int resizeW, resizeH;
    if (aspectRatio > targetAspect) {
        resizeH = targetH; resizeW = static_cast<int>(targetH * aspectRatio);
    } else {
        resizeW = targetW; resizeH = static_cast<int>(targetW / aspectRatio);
    }
    Size resizeSize(resizeW, resizeH);
    auto resized = resize(image, resizeSize);
    int cropX = (resizeW - targetW) / 2;
    int cropY = (resizeH - targetH) / 2;
    auto resizedPtr = resized->readMap<uint8_t>();
    VARP cropped = _Input({targetH, targetW, origC}, NHWC, halide_type_of<uint8_t>());
    auto croppedPtr = cropped->writeMap<uint8_t>();
    for (int h = 0; h < targetH; ++h) {
        int srcRow = (cropY + h) * resizeW * origC;
        int dstRow = h * targetW * origC;
        for (int w = 0; w < targetW; ++w) {
            int srcOff = srcRow + (cropX + w) * origC;
            int dstOff = dstRow + w * origC;
            for (int c = 0; c < origC; ++c) croppedPtr[dstOff + c] = resizedPtr[srcOff + c];
        }
    }
    return cropped;
}

VARP Diffusion::bgrToRgb(VARP bgrImage) {
    auto info = bgrImage->getInfo();
    if (!info || info->dim.size() != 3 || info->dim[2] != 3) return nullptr;
    int H = info->dim[0], W = info->dim[1], C = info->dim[2];
    VARP rgbImage = _Input({H, W, C}, info->order, info->type);
    auto bgrPtr = bgrImage->readMap<uint8_t>();
    auto rgbPtr = rgbImage->writeMap<uint8_t>();
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int idx = (h * W + w) * C;
            rgbPtr[idx + 0] = bgrPtr[idx + 2];
            rgbPtr[idx + 1] = bgrPtr[idx + 1];
            rgbPtr[idx + 2] = bgrPtr[idx + 0];
        }
    }
    return rgbImage;
}

VARP Diffusion::rgbToBgr(VARP rgbImage) { return bgrToRgb(rgbImage); }

VARP Diffusion::hwcToNchw(VARP hwcImage, bool normalize) {
    auto info = hwcImage->getInfo();
    if (!info || info->dim.size() != 3) return nullptr;
    int H = info->dim[0], W = info->dim[1], C = info->dim[2];
    VARP nchwImage = _Input({1, C, H, W}, NCHW, halide_type_of<float>());
    auto srcPtr = hwcImage->readMap<uint8_t>();
    auto dstPtr = nchwImage->writeMap<float>();
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float val = static_cast<float>(srcPtr[(h * W + w) * C + c]);
                if (normalize) val = val / 127.5f - 1.0f;
                dstPtr[c * H * W + h * W + w] = val;
            }
        }
    }
    return nchwImage;
}

VARP Diffusion::nchwToHwc(VARP nchwImage, bool denormalize) {
    auto info = nchwImage->getInfo();
    if (!info || info->dim.size() != 4) return nullptr;
    int C = info->dim[1], H = info->dim[2], W = info->dim[3];
    VARP hwcImage = _Input({H, W, C}, NHWC, halide_type_of<uint8_t>());
    auto srcPtr = nchwImage->readMap<float>();
    auto dstPtr = hwcImage->writeMap<uint8_t>();
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < C; ++c) {
                float val = srcPtr[c * H * W + h * W + w];
                if (denormalize) val = (val + 1.0f) * 127.5f;
                val = std::max(0.0f, std::min(255.0f, val));
                dstPtr[(h * W + w) * C + c] = static_cast<uint8_t>(val + 0.5f);
            }
        }
    }
    return hwcImage;
}

// ===== Latent Packing/Unpacking for Flux-like models =====

void Diffusion::packLatents(const float* src, float* dst, int B, int C, int H, int W, int seqOffset) {
    int pH = H / 2, pW = W / 2;
    for (int b = 0; b < B; ++b) {
        for (int ph = 0; ph < pH; ++ph) {
            for (int pw = 0; pw < pW; ++pw) {
                int seqIdx = seqOffset + ph * pW + pw;
                for (int c = 0; c < C; ++c) {
                    for (int dh = 0; dh < 2; ++dh) {
                        for (int dw = 0; dw < 2; ++dw) {
                            int srcH = ph * 2 + dh, srcW = pw * 2 + dw;
                            int featIdx = c * 4 + dh * 2 + dw;
                            dst[b * (pH * pW + seqOffset) * (C * 4) + seqIdx * (C * 4) + featIdx] =
                                src[b * C * H * W + c * H * W + srcH * W + srcW];
                        }
                    }
                }
            }
        }
    }
}

void Diffusion::unpackLatents(const float* src, float* dst, int B, int C, int H, int W, int seqLen) {
    int pH = H / 2, pW = W / 2;
    for (int b = 0; b < B; ++b) {
        for (int ph = 0; ph < pH; ++ph) {
            for (int pw = 0; pw < pW; ++pw) {
                int seqIdx = ph * pW + pw;
                for (int c = 0; c < C; ++c) {
                    for (int dh = 0; dh < 2; ++dh) {
                        for (int dw = 0; dw < 2; ++dw) {
                            int dstH = ph * 2 + dh, dstW = pw * 2 + dw;
                            int featIdx = c * 4 + dh * 2 + dw;
                            dst[b * C * H * W + c * H * W + dstH * W + dstW] =
                                src[b * seqLen * (C * 4) + seqIdx * (C * 4) + featIdx];
                        }
                    }
                }
            }
        }
    }
}

} // namespace DIFFUSION
} // namespace MNN
