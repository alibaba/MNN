//
//  omni.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <regex>
#include <algorithm>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "omni.hpp"
#include "kvmeta.hpp"
#include "llmconfig.hpp"
#include "tokenizer.hpp"
#include "diskembedding.hpp"
#include "sampler.hpp"
#include "httplib.h"
#ifdef LLM_SUPPORT_VISION
#include <cv/cv.hpp>
#endif
#ifdef LLM_SUPPORT_AUDIO
#include <audio/audio.hpp>
#endif

namespace MNN {
using namespace Express;
namespace Transformer {

template <typename T>
static inline VARP _var(std::vector<T> vec, const std::vector<int> &dims) {
    return _Const(vec.data(), dims, NHWC, halide_type_of<T>());
}

static MNNForwardType backend_type_convert(const std::string& type_str) {
    if (type_str == "cpu")
        return MNN_FORWARD_CPU;
    if (type_str == "metal")
        return MNN_FORWARD_METAL;
    if (type_str == "cuda")
        return MNN_FORWARD_CUDA;
    if (type_str == "opencl")
        return MNN_FORWARD_OPENCL;
    if (type_str == "opengl")
        return MNN_FORWARD_OPENGL;
    if (type_str == "vulkan")
        return MNN_FORWARD_VULKAN;
    if (type_str == "npu")
        return MNN_FORWARD_NN;
    return MNN_FORWARD_AUTO;
}

Omni::Omni(std::shared_ptr<LlmConfig> config) : Llm(config) {
    if (config->is_visual()) {
        mVisionHeight = config->config_.value("image_size", mVisionHeight);
        mVisionWidth  = mVisionHeight;
        mVisionPad    = config->config_.value("image_pad", mVisionPad);
        mVisionStart  = config->config_.value("vision_start", mVisionStart);
        mVisionEnd    = config->config_.value("vision_end", mVisionEnd);
        mVisionMean   = config->config_.value("image_mean", mVisionMean);
        mVisionNorm   = config->config_.value("image_norm", mVisionNorm);
        mVisionSizeUnit = config->config_.value("image_size_unit", mVisionSizeUnit);
        mVisionMaxSize = config->config_.value("image_max_size", mVisionMaxSize);
        mVisionGlobal = config->config_.value("global_image", mVisionGlobal);
    }
    if (config->is_audio()) {}
}

void Omni::load() {
    Llm::load();
    if (mConfig->has_talker()) {
        mTalker.reset(new Talker(mConfig, this));
        mTalker->load();
    }
    ScheduleConfig config;
    if (mConfig->mllm_config_.empty()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        BackendConfig cpuBackendConfig;
        config.type      = backend_type_convert(mConfig->backend_type(true));
        config.numThread = mConfig->thread_num(true);
        if(config.type == 3){
            config.numThread |= 64;
        }
        if (mConfig->power(true) == "high") {
            cpuBackendConfig.power = BackendConfig::Power_High;
        } else if (mConfig->power(true) == "low") {
            cpuBackendConfig.power = BackendConfig::Power_Low;
        }
        if (mConfig->memory(true) == "high") {
            cpuBackendConfig.memory = BackendConfig::Memory_High;
        } else if (mConfig->memory(true) == "low") {
            cpuBackendConfig.memory = BackendConfig::Memory_Low;
        }
        if (mConfig->precision(true) == "high") {
            cpuBackendConfig.precision = BackendConfig::Precision_High;
        } else if (mConfig->precision(true) == "low") {
            cpuBackendConfig.precision = BackendConfig::Precision_Low;
        }
        config.backendConfig = &cpuBackendConfig;
        mProcessorRuntimeManager.reset(Executor::RuntimeManager::createRuntimeManager(config));
        setRuntimeHint(mProcessorRuntimeManager);
    }
    Module::Config module_config;
    if(config.type == MNN_FORWARD_NN) {
        module_config.shapeMutable = false;
        module_config.rearrange    = false;
    } else {
        module_config.shapeMutable = true;
        module_config.rearrange    = true;
    }
    if (mConfig->is_visual()) {
        mVisionModule.reset(Module::load({}, {}, mConfig->visual_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
    if (mConfig->is_audio()) {
        mAudioModule.reset(Module::load({}, {}, mConfig->audio_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
}

#ifdef LLM_SUPPORT_VISION
std::vector<int> Omni::defaultVisionProcess(VARP image) {
    mVisionHeight = UP_DIV(mVisionHeight, mVisionSizeUnit) * mVisionSizeUnit;
    mVisionWidth  = UP_DIV(mVisionWidth, mVisionSizeUnit) * mVisionSizeUnit;
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mVisionMean, mVisionNorm);
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NC4HW4);
    auto imageEmbedding = mVisionModule->forward(image);

    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = imageEmbedding->getInfo()->dim[0];
    std::vector<int> imgIds(visionLen, mVisionPad);
    if (mVisionStart >= 0 && mVisionEnd >= 0) {
        imgIds.insert(imgIds.begin(), mVisionStart);
        imgIds.push_back(mVisionEnd);
    }
    return imgIds;
}

std::vector<int> Omni::qwen2VisionProcess(VARP image) {
    const auto inputNames = mVisionModule->getInfo()->inputNames;
    bool hasWindowIndex = inputNames.size() == 4 && inputNames[3] == "window_index";
    // Qwen2-VL / Qwen2.5-VL
    mVisionHeight = round(mVisionHeight / 28.0) * 28;
    mVisionWidth = round(mVisionWidth / 28.0) * 28;
    image = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                            MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                            mVisionMean, mVisionNorm);
    image = Express::_Unsqueeze(image, {0});
    image = Express::_Convert(image, NCHW);
    auto patches = Express::_Concat({image, image}, 0);
    auto patches_dim = patches->getInfo()->dim;
    int temporal = patches_dim[0];
    int channel  = patches_dim[1];
    int height   = patches_dim[2];
    int width    = patches_dim[3];
    constexpr int temporal_patch_size = 2;
    constexpr int patch_size = 14;
    constexpr int merge_size = 2;
    int grid_t = temporal / temporal_patch_size;
    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    addPositionIds(grid_t, grid_h / merge_size, grid_w / merge_size);
    // build patches
    patches = Express::_Reshape(patches, {
        grid_t, temporal_patch_size,
        channel,
        grid_h / merge_size, merge_size, patch_size,
        grid_w / merge_size, merge_size, patch_size,
    });
    patches = Express::_Permute(patches, {0, 3, 6, 4, 7, 2, 1, 5, 8});
    patches = Express::_Reshape(patches, {
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size
    });
    const int seq_len = grid_t * grid_h * grid_w;
    // build position_ids
    const int wblock_size = merge_size * merge_size;
    const int hblock_size = wblock_size * grid_w / merge_size;
    VARP position_ids = Express::_Input({2, seq_len}, NCHW, halide_type_of<int>());
    auto hpos_ptr = position_ids->writeMap<int>();
    auto wpos_ptr = hpos_ptr + seq_len;
    for (int i = 0; i < grid_h; i++) {
        int h_idx = i / merge_size, h_off = i % merge_size;
        for (int j = 0; j < grid_w; j++) {
            int w_idx = j / merge_size, w_off = j % merge_size;
            int index = h_idx * hblock_size + w_idx * wblock_size + h_off * 2 + w_off;
            hpos_ptr[index] = i;
            wpos_ptr[index] = j;
        }
    }
    VARP attention_mask, window_index;
    VARPS moduleInputs= {patches, position_ids};
    if (hasWindowIndex) {
        // build window_index
        window_index = Express::_Input({seq_len / 4}, NCHW, halide_type_of<int>());
        auto window_index_ptr = window_index->writeMap<int>();
        const int merge_unit = merge_size * merge_size;
        const int vit_merger_window_size = 4;
        int llm_grid_h = grid_h / merge_size;
        int llm_grid_w = grid_w / merge_size;
        int pad_h = vit_merger_window_size - (llm_grid_h % vit_merger_window_size);
        int pad_w = vit_merger_window_size - (llm_grid_w % vit_merger_window_size);
        int new_h = llm_grid_h + pad_h;
        int new_w = llm_grid_w + pad_w;
        int num_windows_h = new_h / vit_merger_window_size;
        int num_windows_w = new_w / vit_merger_window_size;
        std::vector<int> seqlens;
        int window_index_idx = 0;
        for (int t = 0; t < grid_t; ++t) {
            for (int win_h = 0; win_h < num_windows_h; ++win_h) {
                for (int win_w = 0; win_w < num_windows_w; ++win_w) {
                    int count = 0;
                    for (int i = 0; i < vit_merger_window_size; ++i) {
                        int h_global = win_h * vit_merger_window_size + i;
                        if (h_global >= llm_grid_h) continue;
                        for (int j = 0; j < vit_merger_window_size; ++j) {
                            int w_global = win_w * vit_merger_window_size + j;
                            if (w_global >= llm_grid_w) continue;
                            int idx = t * llm_grid_h * llm_grid_w + h_global * llm_grid_w + w_global;
                            window_index_ptr[window_index_idx++] = idx;
                            ++count;
                        }
                    }
                    seqlens.push_back(count);
                }
            }
        }
        std::vector<int> cu_window_seqlens = {0};
        int prev = cu_window_seqlens.back();
        for (int s : seqlens) {
            cu_window_seqlens.push_back(prev + s * merge_unit);
            prev = cu_window_seqlens.back();
        }
        // build attention_mask
        attention_mask = Express::_Input({2, 1, seq_len, seq_len}, NCHW);
        auto attention_mask_ptr = attention_mask->writeMap<float>();
        ::memset(attention_mask_ptr, 0, seq_len * seq_len * sizeof(float));
        attention_mask_ptr = attention_mask_ptr + seq_len * seq_len;
        for (int i = 0; i < seq_len * seq_len; i++) {
            attention_mask_ptr[i] = std::numeric_limits<float>::lowest();
        }
        for (size_t i = 1; i < cu_window_seqlens.size(); ++i) {
            for (int j = cu_window_seqlens[i - 1]; j < cu_window_seqlens[i]; ++j) {
                for (int k = cu_window_seqlens[i - 1]; k < cu_window_seqlens[i]; ++k) {
                    attention_mask_ptr[seq_len * j + k] = 0;
                }
            }
        }
        moduleInputs.push_back(attention_mask);
        moduleInputs.push_back(window_index);
    } else {
        // build attention_mask
        attention_mask = Express::_Input({1, seq_len, seq_len}, NCHW);
        ::memset(attention_mask->writeMap<float>(), 0, seq_len * seq_len * sizeof(float));
        moduleInputs.push_back(attention_mask);
    }
#ifdef DEBUG_IMAGE
    patches.fix(MNN::Express::VARP::CONSTANT);
    patches->setName("patches");
    position_ids.fix(MNN::Express::VARP::CONSTANT);
    position_ids->setName("position_ids");
    attention_mask.fix(MNN::Express::VARP::CONSTANT);
    attention_mask->setName("attention_mask");
    MNN::Express::Variable::save({patches, position_ids, attention_mask}, "input.mnn");
#endif
    auto imageEmbedding = mVisionModule->onForward(moduleInputs)[0];
#ifdef DEBUG_IMAGE
    imageEmbedding->setName("image_embeds");
    MNN::Express::Variable::save({imageEmbedding}, "output.mnn");
#endif
    mVisionEmbeddings.push_back(imageEmbedding);
    int visionLen = imageEmbedding->getInfo()->dim[0];
    std::vector<int> imgIds(visionLen, mVisionPad);
    imgIds.insert(imgIds.begin(), mVisionStart);
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

std::vector<int> Omni::smolvlmVisionProcess(VARP image) {
    // SmolVLM
    constexpr int visionLen = 64;
    bool splitImage = mVisionHeight > mVisionSizeUnit || mVisionWidth > mVisionSizeUnit;
    auto globalImage = MNN::CV::resize(image, {mVisionSizeUnit, mVisionSizeUnit}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
    globalImage = Express::_Unsqueeze(globalImage, {0});
    globalImage = Express::_Convert(globalImage, NCHW);
    std::vector<int> imgIds;
    if (splitImage) {
        mVisionHeight = round(mVisionHeight / (float)mVisionSizeUnit) * mVisionSizeUnit;
        mVisionWidth = round(mVisionWidth / (float)mVisionSizeUnit) * mVisionSizeUnit;
        if (mVisionHeight > mVisionMaxSize) {
            mVisionHeight = mVisionMaxSize;
        }
        if (mVisionWidth > mVisionMaxSize) {
            mVisionWidth = mVisionMaxSize;
        }
        auto patches = MNN::CV::resize(image, {mVisionWidth, mVisionHeight}, 0, 0,
                                       MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                       mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        auto imageDims = patches->getInfo()->dim;
        int batch    = imageDims[0];
        int channel  = imageDims[1];
        int height   = imageDims[2];
        int width    = imageDims[3];
        int grid_h = height / mVisionSizeUnit;
        int grid_w = width / mVisionSizeUnit;
        patches = Express::_Reshape(patches, {
            batch,
            channel,
            grid_h, mVisionSizeUnit,
            grid_w, mVisionSizeUnit,
        });
        patches = Express::_Permute(patches, {0, 2, 4, 1, 3, 5});
        patches = Express::_Reshape(patches, {
            batch * grid_h * grid_w,
            channel,
            mVisionSizeUnit,
            mVisionSizeUnit
        });
        patches = _Concat({patches, globalImage}, 0);
        auto imageEmbedding = mVisionModule->forward(patches);
        auto embeddingDims = imageEmbedding->getInfo()->dim;
        for (int i = 0; i < embeddingDims[0]; i++) {
            auto embedding = _Squeeze(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {0});
            mVisionEmbeddings.push_back(embedding);
        }
        int endRow = tokenizer_encode("\n")[0];
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                imgIds.push_back(mVisionStart);
                // <row_{h+1}_col{w+1}>
                std::string image_pos = "<row_" + std::to_string(h + 1) + "_col_" + std::to_string(w + 1) + ">";
                imgIds.push_back(tokenizer_encode(image_pos)[0]);
                for (int p = 0; p < visionLen; p++) {
                    imgIds.push_back(mVisionPad);
                }
            }
            imgIds.push_back(endRow);
        }
        imgIds.push_back(endRow);
    } else {
        auto imageEmbedding = mVisionModule->forward(globalImage);
        mVisionEmbeddings.push_back(_Squeeze(imageEmbedding, {0}));
    }
    // global image ids
    imgIds.push_back(mVisionStart);
    imgIds.push_back(mVisionGlobal);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    return imgIds;
}

std::vector<std::pair<int, int>> minicpmBestSize(std::pair<int, int> original_size, int patch_size) {
    constexpr int max_slice_nums = 9, scale_resolution = 448;
    auto _get_target_size =
        [&](std::pair<int, int> size, bool upscale) -> std::pair<int, int> {
        int h = size.first;
        int w = size.second;
        int target_w, target_h;
        if (!upscale && (static_cast<long long>(w) * h <= static_cast<long long>(scale_resolution) * scale_resolution)) {
            target_w = w;
            target_h = h;
        } else {
            double r = (h != 0) ? static_cast<double>(w) / h : 0.0;
            if (r > 0) {
                target_h = static_cast<int>(scale_resolution / std::sqrt(r));
                target_w = static_cast<int>(target_h * r);
            } else {
                target_h = 0;
                target_w = scale_resolution;
            }
        }
        int final_h = std::max(static_cast<int>(std::round(static_cast<double>(target_h) / patch_size)) * patch_size, patch_size);
        int final_w = std::max(static_cast<int>(std::round(static_cast<double>(target_w) / patch_size)) * patch_size, patch_size);
        return std::make_pair(final_h, final_w);
    };
    int original_height = original_size.first;
    int original_width = original_size.second;
    double ratio = (static_cast<double>(original_width) * original_height) / (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    std::vector<std::pair<int, int>> candidates;
    std::set<int> nums_to_check;
    if (multiple > 1) nums_to_check.insert(multiple - 1);
    nums_to_check.insert(multiple);
    nums_to_check.insert(multiple + 1);
    for (std::set<int>::iterator it = nums_to_check.begin(); it != nums_to_check.end(); ++it) {
        int num = *it;
        if (num >= 1 && num <= max_slice_nums) {
            for (int m = 1; m * m <= num; ++m) {
                if (num % m == 0) {
                    candidates.push_back(std::make_pair(m, num / m));
                    if (m * m != num) candidates.push_back(std::make_pair(num / m, m));
                }
            }
        }
    }
    if (candidates.empty()) { candidates.push_back(std::make_pair(1, 1)); }
    double log_ratio = std::log(static_cast<double>(original_width) / original_height);
    std::pair<int, int> best_grid = *std::min_element(candidates.begin(), candidates.end(),
        [log_ratio](const std::pair<int, int>& g1, const std::pair<int, int>& g2) {
            auto key = [log_ratio](const std::pair<int, int>& g) -> double {
                if (g.first == 0) return std::numeric_limits<double>::infinity();
                return std::abs(log_ratio - std::log(static_cast<double>(g.second) / g.first));
            };
            return key(g1) < key(g2);
        });
    std::pair<int, int> source_image_size = _get_target_size(original_size, false);
    double patch_h = static_cast<double>(original_height) / best_grid.first;
    double patch_w = static_cast<double>(original_width) / best_grid.second;
    std::pair<int, int> best_patch_size = _get_target_size(std::make_pair(static_cast<int>(patch_h), static_cast<int>(patch_w)), true);
    std::pair<int, int> refine_image_size = std::make_pair(
        best_patch_size.first * best_grid.first,
        best_patch_size.second * best_grid.second
    );
    std::vector<std::pair<int, int>> result;
    result.push_back(source_image_size);
    result.push_back(refine_image_size);
    result.push_back(best_grid);
    return result;
}

std::vector<int> Omni::minicpmVisionProcess(VARP image) {
    constexpr int visionLen = 64, patchesPerSide = 70;
    const int patchSize = mVisionSizeUnit;
    auto bestSize = minicpmBestSize(std::make_pair(mVisionHeight, mVisionWidth), patchSize);
    auto globalSize = bestSize[0];
    auto refineSize = bestSize[1];
    auto sliceGrids = bestSize[2];
    auto reoderImage = [this, &patchSize](
        Express::VARP img, std::pair<int, int> targetSize, std::pair<int,int> grid, std::vector<int>& tgtSize) {
        auto patches = MNN::CV::resize(img, {targetSize.second, targetSize.first}, 0, 0,
                                    MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                    mVisionMean, mVisionNorm);
        patches = Express::_Unsqueeze(patches, {0});
        patches = Express::_Convert(patches, NCHW);
        auto imageDims = patches->getInfo()->dim;
        int batch   = imageDims[0];
        int channel = imageDims[1];
        int height  = imageDims[2];
        int width   = imageDims[3];
        int gridH   = grid.first;
        int gridW   = grid.second;
        int subHeight = height / gridH;
        int subWidth = width / gridW;
        int numPatchesH = subHeight / patchSize;
        int numPatchesW = subWidth / patchSize;
        patches = Express::_Reshape(patches, {
            channel,
            gridH,
            numPatchesH,
            patchSize,
            gridW,
            numPatchesW,
            patchSize
        });
        patches = Express::_Permute(patches, {1, 4, 0, 3, 2, 5, 6});
        patches = Express::_Reshape(patches, {
            gridH * gridW,
            channel,
            patchSize,
            numPatchesH * numPatchesW * patchSize
        });
        for (int i = 0; i < gridH * gridW; i++) {
            tgtSize.push_back(numPatchesH);
            tgtSize.push_back(numPatchesW);
        }
        return patches;
    };
    // pixel values
    std::vector<int> tgtSize;
    auto globalImage = reoderImage(image, globalSize, std::make_pair(1, 1), tgtSize);
    auto refineImage = reoderImage(image, refineSize, sliceGrids, tgtSize);
    int globleDim = globalImage->getInfo()->dim[3];
    int refineDim = refineImage->getInfo()->dim[3];
    globalImage = _Pad(globalImage, _var<int>({0, 0, 0, 0, 0, 0, 0, refineDim - globleDim}, {8}), CONSTANT);
    auto pixel_values = _Concat({globalImage, refineImage}, 0);
    // position ids
    int B = tgtSize.size() / 2;
    int S = tgtSize[0] * tgtSize[1];
    int L = tgtSize[2] * tgtSize[3];
    auto position_ids = Express::_Input({B, L}, NCHW, halide_type_of<int>());
    auto posPtr = position_ids->writeMap<int>();
    memset(posPtr, 0, B * L * sizeof(int));
    for (int i = 0; i < B; ++i) {
        int nb_patches_h = tgtSize[i * 2];
        int nb_patches_w = tgtSize[i * 2 + 1];
        for (int h_idx = 0; h_idx < nb_patches_h; ++h_idx) {
            long bucket_h = static_cast<long>(std::floor(
                (static_cast<float>(h_idx) / nb_patches_h) * patchesPerSide
            ));
            for (int w_idx = 0; w_idx < nb_patches_w; ++w_idx) {
                long bucket_w = static_cast<long>(std::floor(
                    (static_cast<float>(w_idx) / nb_patches_w) * patchesPerSide
                ));
                long pos_id = bucket_h * patchesPerSide + bucket_w;
                long patch_idx = h_idx * nb_patches_w + w_idx;
                posPtr[i * L + patch_idx] = static_cast<int>(pos_id);
            }
        }
    }
    // attention mask
    auto attention_mask = Express::_Input({B, L}, NCHW);
    auto maskPtr = attention_mask->writeMap<float>();
    memset(maskPtr, 0, B * L * sizeof(float));
    for (int i = S; i < L; i++) {
        maskPtr[i] = std::numeric_limits<float>::lowest();
    }
    // tgt size
    auto tgt_sizes = Express::_Input({B, 2}, NCHW, halide_type_of<int>());
    ::memcpy(tgt_sizes->writeMap<int>(), tgtSize.data(), tgtSize.size() * sizeof(int));
    auto imageEmbedding = mVisionModule->onForward({pixel_values, position_ids, attention_mask, tgt_sizes})[0];
    for (int i = 0; i < B; i++) {
        auto embedding = _Permute(_GatherV2(imageEmbedding, _var<int>({i}, {1}), _var<int>({0}, {1})), {1, 0, 2});
        mVisionEmbeddings.push_back(embedding);
    }
    int visionSliceStart = mConfig->config_.value("vision_slice_start_id", 111);
    int visionSliceEnd = mConfig->config_.value("vision_slice_end_id", 112);
    int visionIdStart = mConfig->config_.value("vision_id_start_id", 113);
    int visionIdEnd = mConfig->config_.value("vision_id_end_id", 114);
    std::vector<int> imgIds;
    // image id
    imgIds.push_back(visionIdStart);
    auto visionIdxIds = tokenizer_encode(std::to_string(mVisionNum));
    for (auto idx : visionIdxIds) {
        imgIds.push_back(idx);
    }
    imgIds.push_back(visionIdEnd);
    // global image
    imgIds.push_back(mVisionStart);
    for (int p = 0; p < visionLen; p++) {
        imgIds.push_back(mVisionPad);
    }
    imgIds.push_back(mVisionEnd);
    // slice images
    for (int i = 0; i < B - 1; i++) {
        imgIds.push_back(visionSliceStart);
        for (int p = 0; p < visionLen; p++) {
            imgIds.push_back(mVisionPad);
        }
        imgIds.push_back(visionSliceEnd);
    }
    return imgIds;
}
#endif

std::vector<int> Omni::visionProcess(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);
    return visionProcess(image);
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::visionProcess(VARP image) {
#ifdef LLM_SUPPORT_VISION
    if (image == nullptr) {
        MNN_PRINT("Omni Can't open image\n");
        return std::vector<int>(0);
    }
    Timer _t;
    std::vector<int> imgIds;
    const auto inputNames = mVisionModule->getInfo()->inputNames;
    if (inputNames.size() >= 3 && inputNames[0] == "patches") {
        imgIds = qwen2VisionProcess(image);
    } else if (inputNames[0] == "pixel_values") {
        if (inputNames.size() == 1) {
            imgIds = smolvlmVisionProcess(image);
        } else {
            imgIds = minicpmVisionProcess(image);
        }
    } else {
        imgIds = defaultVisionProcess(image);
    }
    mContext->vision_us += _t.durationInUs();
    // set vision number for image idx
    mVisionNum += 1;
    return imgIds;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::audioProcess(const std::string& file) {
#ifdef LLM_SUPPORT_AUDIO
    constexpr int sample_rate = 16000;
    auto load_res        = MNN::AUDIO::load(file, sample_rate);
    VARP waveform        = load_res.first;
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't open audio: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    return audioProcess(waveform);
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::audioProcess(MNN::Express::VARP waveform) {
#ifdef LLM_SUPPORT_AUDIO
    if (waveform == nullptr) {
        MNN_PRINT("Omni Can't process audio: waveform is null\n");
        return std::vector<int>(0);
    }
    
    Timer _t;
    auto input_features  = MNN::AUDIO::whisper_fbank(waveform);
    VARP audio_embedding;
    if (mAudioModule->getInfo()->inputNames.size() > 1) {
        int seqlen = UP_DIV(input_features->getInfo()->dim[2], 2);
        constexpr int n_window = 100;
        std::vector<int> cu_seqlens;
        int curseq = 0;
        while (curseq < seqlen) {
            cu_seqlens.push_back(curseq);
            curseq += n_window;
        }
        if (seqlen % n_window != 0) {
            cu_seqlens.push_back(seqlen);
        }
        VARP attention_mask = _Input({1, seqlen, seqlen}, NCHW, halide_type_of<float>());
        auto ptr = attention_mask->writeMap<float>();
        for (int i = 0; i < seqlen; i++) {
            for (int j = 0; j < seqlen; j++) {
                ptr[seqlen * i + j] = std::numeric_limits<float>::lowest();
            }
        }
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            for (int j = cu_seqlens[i - 1]; j < cu_seqlens[i]; ++j) {
                for (int k = cu_seqlens[i - 1]; k < cu_seqlens[i]; ++k) {
                    ptr[seqlen * j + k] = 0;
                }
            }
        }
        audio_embedding = mAudioModule->onForward({input_features, attention_mask})[0];
    } else {
        // Qwen2-Audio just support audio time <= 30s
        if (input_features->getInfo()->dim[2] > 3000) {
            input_features = _Slice(input_features, _var<int>({0, 0, 0}, {3}), _var<int>({-1, -1, 3000}, {3}));
        }
        audio_embedding = mAudioModule->forward(input_features);
    }

    audio_embedding = _Permute(audio_embedding, {1, 0, 2});
    mContext->audio_us = _t.durationInUs();
    mAudioEmbeddings.push_back(audio_embedding);
    int embed_len = audio_embedding->getInfo()->dim[0];
    addPositionIds(embed_len);
    std::vector<int> audio_ids(embed_len, mAudioPad);
    return audio_ids;
#else
    return std::vector<int>(0);
#endif
}

std::vector<int> Omni::multimodeProcess(const std::string& mode, std::string info) {
    auto file_info = info;
    if (mode == "img") {
        std::regex hw_regex(R"(<hw>(.*?)</hw>)");
        std::sregex_iterator iter(info.begin(), info.end(), hw_regex);
        std::sregex_iterator end;
        file_info = "";

        size_t currentPosition = 0;
        if (iter != end) {
            std::smatch match = *iter;
            size_t matchPosition = match.position();
            if (matchPosition > currentPosition) {
                file_info.append(info.substr(currentPosition, matchPosition - currentPosition));
            }

            std::stringstream hw_ss(match.str(1));
            char comma;
            hw_ss >> mVisionHeight >> comma >> mVisionWidth;
            currentPosition = matchPosition + match.length();
        }
        if (currentPosition < info.length()) {
            file_info.append(info.substr(currentPosition));
        }
        // std::cout << "hw: " << mVisionHeight << ", " << mVisionWidth << std::endl;
        // std::cout << "file: " << file_info << std::endl;
    }
    if (file_info.substr(0, 4) == "http") {
        std::regex url_regex(R"(^https?://([^/]+)(/.*))");
        std::smatch url_match_result;
        std::string host, path;
        if (std::regex_search(file_info, url_match_result, url_regex) && url_match_result.size() == 3) {
            host = url_match_result[1].str();
            path = url_match_result[2].str();
        }
        // std::cout << host << "#" << path << std::endl;
        httplib::Client cli(host);
        auto res  = cli.Get(path);
        file_info = "downloaded_file";
        if (res && res->status == 200) {
            std::ofstream file(file_info, std::ios::binary);
            if (file.is_open()) {
                file.write(res->body.c_str(), res->body.size());
                std::cout << "File has been downloaded successfully." << std::endl;
                file.close();
            } else {
                std::cerr << "Unable to open file to write." << std::endl;
            }
        } else {
            std::cerr << "Failed to download file. Status code: " << (res ? res->status : 0) << std::endl;
        }
    }
    if (mode == "img" && mConfig->is_visual()) {
        return visionProcess(file_info);
    }
    if (mode == "audio" && mConfig->is_audio()) {
        return audioProcess(file_info);
    }
    return std::vector<int>(0);
}

void Omni::addPositionIds(int t, int h, int w) {
    int cur_idx = mPositionIds.currentIdx();
    if (h < 0 && w < 0) { // text position ids
        for (int i = 0; i < t; i++) {
            int idx = cur_idx + i;
            mPositionIds.push_back(idx);
        }
    } else { // vision position ids
        // vision start
        mPositionIds.push_back(cur_idx++);
        for (int t_i = 0; t_i < t; t_i++) {
            for (int h_i = 0; h_i < h; h_i++) {
                for (int w_i = 0; w_i < w; w_i++) {
                    mPositionIds.push_back(cur_idx + t_i, cur_idx + h_i, cur_idx + w_i);
                }
            }
        }
        // vision end
        mPositionIds.push_back();
    }
}

std::vector<int> Omni::tokenizer_encode(const MultimodalPrompt& multimodal_input) {
    std::string prompt = multimodal_input.prompt_template;
    // MNN_PRINT("tokenizer_encode(MultimodalPrompt) prompt: %s", prompt.c_str());
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<int> ids{};
    mPositionIds.clear();
    
    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        auto txt_ids = mTokenizer->encode(match.prefix().str());
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        std::string mode = match[1].str();
        std::string content = match[2].str();
        std::vector<int> mul_ids;
        if (mode == "img") {
            mul_ids = processImageContent(content, multimodal_input.images);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) image mul_ids size: %lu", mul_ids.size());
        } else if (mode == "audio") {
            mul_ids = processAudioContent(content, multimodal_input.audios);
            // MNN_PRINT("tokenizer_encode(MultimodalPrompt) audio mul_ids size: %lu", mul_ids.size());
        }
        
        ids.insert(ids.end(), mul_ids.begin(), mul_ids.end());
        searchStart = match.suffix().first;
    }
    if (searchStart != prompt.cend()) {
        auto txt_ids = mTokenizer->encode(std::string(searchStart, prompt.cend()));
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
    }
    return ids;
}

std::vector<int> Omni::tokenizer_encode(const std::string& prompt) {
    MultimodalPrompt multimodal_input;
    multimodal_input.prompt_template = prompt;
    return tokenizer_encode(multimodal_input);
}

std::vector<int> Omni::processImageContent(const std::string& content, const std::map<std::string, PromptImagePart>& images) {
    auto it = images.find(content);
    if (it != images.end()) {
        if (it->second.height > 0 && it->second.width > 0) {
            mVisionHeight = it->second.height;
            mVisionWidth = it->second.width;
        }
        // MNN_PRINT("processImageContent: using placeholder '%s' with size %dx%d", content.c_str(), mVisionWidth, mVisionHeight);
        return visionProcess(it->second.image_data);
    }
    // MNN_PRINT("processImageContent: treating '%s' as file path or URL", content.c_str());
    return multimodeProcess("img", content);
}

std::vector<int> Omni::processAudioContent(const std::string& content, const std::map<std::string, PromptAudioPart>& audios) {
    auto it = audios.find(content);
    if (it != audios.end()) {
        // MNN_PRINT("processAudioContent: using placeholder '%s'", content.c_str());
        if (it->second.waveform.get() != nullptr) {
            return audioProcess(it->second.waveform);
        } else if (!it->second.file_path.empty()) {
            return audioProcess(it->second.file_path);
        } else {
            MNN_PRINT("processAudioContent: audio_part has no valid input\n");
            return std::vector<int>(0);
        }
    }
    // MNN_PRINT("processAudioContent: treating '%s' as file path", content.c_str());
    return multimodeProcess("audio", content);
}

VARP Omni::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    std::vector<int> position_ids;
    int vision_idx = 0, audio_idx = 0;
    std::vector<int> cur_txt_ids;
    bool inVision = false, inAudio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (inAudio) {
            if (id == mAudioPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inAudio = false;
            }
        } else if (id == mAudioPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mAudioEmbeddings[audio_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inAudio = true;
        }
        // vision
#if 1
        if (inVision) {
            if (id == mVisionPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                inVision = false;
            }
        } else if (id == mVisionPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mVisionEmbeddings[vision_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            inVision = true;
        }
        cur_txt_ids.push_back(id);
#else
        if (id == mVisionPad) {
            continue;
        }
        cur_txt_ids.push_back(id);
        if (id == mVisionStart) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mVisionEmbeddings[vision_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
        } else if (id == mVisionEnd) {
            cur_txt_ids.clear();
            cur_txt_ids.push_back(id);
        }
#endif
    }

    mVisionEmbeddings.clear();
    mAudioEmbeddings.clear();
    if (!cur_txt_ids.empty()) {
        auto txt_embedding = Llm::embedding(cur_txt_ids);
        embeddings.push_back(txt_embedding);
    }
    auto embedding = Express::_Concat(embeddings, 0);
    return embedding;
}

static inline bool needNewVar(VARP var, int axis, int seq_len) {
    if (var == nullptr) {
        return true;
    }
    if (var->getInfo()->dim[axis] != seq_len) {
        return true;
    }
    return false;
}

VARP Omni::gen_position_ids(int seq_len) {
    auto positionIdsDims = mModules[0]->getInfo()->inputs[2].dim;
    if (positionIdsDims[0] == 1) {
        return Llm::gen_position_ids(seq_len);
    }
    // mrope
    if (needNewVar(positionIds, 1, seq_len)) {
        positionIds = _Input({3, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (mContext->gen_seq_len > 0) {
        for (int i=0; i<seq_len; ++i) {
            // auto pos = mContext->gen_seq_len + mPositionIds.back() + i;
            auto pos = mContext->all_seq_len + i;
            ptr[i + 0] = pos;
            ptr[i + seq_len] = pos;
            ptr[i + seq_len * 2] = pos;
        }
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i] + mContext->all_seq_len;
            ptr[i + seq_len] = mPositionIds.mH[i] + mContext->all_seq_len;
            ptr[i + seq_len * 2] = mPositionIds.mW[i] + mContext->all_seq_len;
        }
        if (mTalker) {
            mTalker->setPostionIds(mPositionIds);
        }
    }
    // // dump position ids
    // printf("position_ids = [");
    // for (int i = 0; i < seq_len; i++) {
    //     printf("%d ", ptr[i]);
    // }
    // printf("]\n");
    return positionIds;
}

std::vector<Express::VARP> Omni::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    auto outputs = Llm::forwardRaw(hiddenState, mask, inputPos);
    if (mTalker && outputs.size() > 1) {
        mTalker->addTalkerEmbeds(outputs[1]);
    }
    return outputs;
}

void Omni::response(const std::vector<int>& input_ids, std::ostream* os, const char* end_with, int max_new_tokens) {
    if (!end_with) { end_with = "\n"; }
    generate_init(os, end_with);
    if (mTalker) {
        mTalker->generate_init();
    }
    generate(input_ids, max_new_tokens);
}

void Omni::setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) {
    if (mTalker) {
        mTalker->setWavformCallback(callback);
    }
}

void Omni::generateWavform() {
    if (mTalker) {
        mTalker->generate();
#ifdef DUMP_TALKER_PERFORMANCE
        auto context = mTalker->getContext();
        float prefill_s = context->prefill_us / 1e6;
        float decode_s = context->decode_us / 1e6;
        float token2wav_s = context->audio_us / 1e6;
        float dit_s = context->vision_us / 1e6;
        float tts_s = token2wav_s;
        if (mTalker->mStreamWithDecode) {
            tts_s += decode_s;
        }
        float audio_duration = context->gen_seq_len / 50.0;
        printf("\n#################################\n");
        printf("prompt tokens num = %d\n", context->prompt_len);
        printf("decode tokens num = %d\n", context->gen_seq_len);
        printf("  prefill time = %.2f s\n", prefill_s);
        printf("   decode time = %.2f s\n", decode_s);
        printf("      dit time = %.2f s\n", dit_s);
        printf("token2wav time = %.2f s\n", token2wav_s);
        printf("      tts time = %.2f s\n", tts_s);
        printf("  prefill speed = %.2f tok/s\n", context->prompt_len / prefill_s);
        printf("   decode speed = %.2f tok/s\n", context->gen_seq_len / decode_s);
        printf("token2wav speed = %.2f tok/s\n", context->gen_seq_len / token2wav_s);
        printf("      tts rtf   = %.2f \n", tts_s / audio_duration);
        printf("##################################\n");
#endif
    }
}

void Talker::load() {
    initRuntime();
    mSeqLenIndex = 1;
    set_config("{\"sampler_type\": \"mixed\", \"temperature\": 0.9, \"topK\": 40, \"topP\": 0.8, \"penalty\": 1.05}");
    mSampler.reset(Sampler::createSampler(mContext, mConfig));
    mDiskEmbedding.reset(new DiskEmbedding(mConfig, mConfig->talker_embedding_file()));
    // some embeddings
    mMaxNewTokens = mConfig->talker_max_new_tokens();
    std::string speaker = mConfig->talker_speaker();
    auto spk_dict = Express::Variable::loadMap(mConfig->spk_dict().c_str());
    mSpk = spk_dict[speaker + "_spk"];
    mCond = spk_dict[speaker + "_cond"];
    mTextBosToken = int(spk_dict[speaker + "_bos_token"]->readMap<float>()[0]);
    mTextBos = mThinker->embedding({mTextBosToken});
    mTextEos = mThinker->embedding({mTextEosToken});
    mTextPad = mThinker->embedding({mTextPadToken});
    mCodecBos = embedding({mCodecBosToken});
    mCodecPad = embedding({mCodecPadToken});

    Module::Config module_config;
    module_config.shapeMutable = false;
    module_config.rearrange    = true;
    mModules.resize(1);
    std::vector<std::string> inputNames {"inputs_embeds", "attention_mask", "position_ids", "logits_index"};

    mModules[0].reset(Module::load(inputNames,
                                    {"logits"}, mConfig->talker_model().c_str(), mRuntimeManager, &module_config));
    // dit
    mPreDit.reset(Module::load({"cond", "spk", "code"}, {"code_embeds", "rope", "mask"},
                                mConfig->predit_model().c_str(), mRuntimeManager, &module_config));
    mDit.reset(Module::load({"x", "code_embeds", "rope", "mask", "time"}, {"mel"},
                            mConfig->dit_model().c_str(), mRuntimeManager, &module_config));
    // bigvgan
    mBigvgan.reset(Module::load({"generated_mel"},
                                {"waveform"}, mConfig->bigvgan_model().c_str(), mRuntimeManager, &module_config));
    // autoregressive decode module
    mModulePool[std::make_pair(1, false)].reset(Module::clone(mModules[0].get()));
    // prefill module
    mModulePool[std::make_pair(mPrefillKey, mConfig->all_logits())] = mModules[0];
}

void Talker::generate_init(std::ostream* os, const char* end_with) {
    if (!doGenerate()) { return; }
    Llm::generate_init(os, end_with);
    // stream generate init
    mTalkerEmbeds.clear();
    if (mInitialNoise.empty()) {
        mInitialNoise.resize(mMaxNewTokens * 2 * 80);
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < mMaxNewTokens * 2 * 80; ++i) {
            mInitialNoise[i] = distribution(generator);
        }
    }
    mWaveformBuffer.reserve(mMaxNewTokens * 2 * 240);
    mMelBuffer = nullptr;
    dit_start_index = 0;
    dit_left_padding = 0;
    vocoder_left_pad = 0;
}

Express::VARP Talker::embedding(const std::vector<int>& input_ids) {
    return Llm::embedding(input_ids);
}

Express::VARP Talker::gen_position_ids(int seq_len) {
    // mrope
    if (needNewVar(positionIds, 2, seq_len)) {
        positionIds = _Input({3, 1, seq_len}, NCHW, halide_type_of<int>());
    }
    auto ptr = positionIds->writeMap<int>();
    if (seq_len == 1) {
        ptr[0] = mContext->gen_seq_len + mPositionIds.back();
        ptr[1] = ptr[0];
        ptr[2] = ptr[0];
    } else {
        for (int i = 0; i < seq_len; i++) {
            ptr[i] = mPositionIds.mT[i];
            ptr[i + seq_len] = mPositionIds.mH[i];
            ptr[i + seq_len * 2] = mPositionIds.mW[i];
        }
    }
    return positionIds;
}

void Talker::setWavformCallback(const std::function<bool(const float*, size_t, bool)> callback) {
    mWavformCallback = callback;
}

VARP Talker::ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise) {
    auto code = _Const(codec_tokens, {1, codec_size}, NCHW, halide_type_of<int>());
    const int max_duration = codec_size * 2;
    auto outputs = mPreDit->onForward({mCond, mSpk, code});
    auto code_embeds = outputs[0];
    auto rope = outputs[1];
    auto mask = outputs[2];
    const int steps = mConfig->dit_steps();
    const int solver = mConfig->dit_solver();
    const float step_ratio = 1.0 / (steps - 1);
    auto forward_dit = [&](float t, Express::VARP x) {
        auto pred = mDit->onForward({x, code_embeds, rope, mask, _Const(t, {1}, NCHW)})[0];
        return pred;
    };
    auto y0 = _Input({1, max_duration, 80}, NCHW, halide_type_of<float>());
    if (initial_noise) {
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = initial_noise[i];
        }
    } else {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < max_duration * 80; ++i) {
            y0->writeMap<float>()[i] = distribution(generator);
        }
    }
    MNN::Timer _t;
    for (int i = 0; i < steps - 1; i++) {
        float t0 = 1 - std::cos(M_PI / 2 * i * step_ratio);
        float t1 = 1 - std::cos(M_PI / 2 * (i + 1) * step_ratio);
        float dt = t1 - t0;
        auto k1 = mDit->onForward({y0, code_embeds, rope, mask, _Const(t0, {1}, NCHW)})[0];
        if (solver == 1) {
            y0 = y0 + k1 * _Scalar<float>(dt);
        } else {
            constexpr float one_third = 1.0 / 3.0;
            constexpr float two_third = 2.0 / 3.0;
            auto kk1 = _Clone(k1, true);
            auto k2 = forward_dit(t0 + dt * one_third, y0 + k1 * _Scalar<float>(dt * one_third));
            auto kk2 = _Clone(k2, true);
            auto k3 = forward_dit(t0 + dt * two_third, y0 + _Scalar<float>(dt) * (k2 - k1 * _Scalar<float>(two_third)));
            auto kk3 = _Clone(k3, true);
            auto k4 = forward_dit(t1, y0 + _Scalar<float>(dt) * (k1 - k2 + k3));
            auto kk4 = _Clone(k4, true);
            auto dy = (kk1 + _Scalar<float>(3.0) * (kk2 + kk3) + kk4) * _Scalar<float>(dt * 0.125);
            y0 = y0 + dy;
        }
    }
    mContext->vision_us += _t.durationInUs();
    auto generated_mel = _Permute(y0, {0, 2, 1});
    return generated_mel;
}

VARP Talker::bigvganForward(VARP mel) {
    auto waveform = mBigvgan->forward(mel);
    return waveform;
}

void Talker::token2wav(bool talker_done) {
    int codec_size = mContext->gen_seq_len - dit_start_index;
    int chunk_size = dit_left_padding + dit_chunk_size + dit_right_padding;
    bool last_chunk = talker_done && (codec_size <= chunk_size);
    // prefill some codec tokens
    // if (!talker_done && mMelBuffer == nullptr && codec_size < chunk_size * 2) {
    //     return;
    // }
    if (!last_chunk && codec_size < chunk_size) {
        return;
    }
    auto codec_ptr = mContext->output_tokens.data() + dit_start_index;
    auto noise_ptr = mInitialNoise.data() + dit_start_index * 160;
    int real_size = last_chunk ? codec_size : chunk_size;
    int mel_size = last_chunk ? -1 : dit_chunk_size * 2;
    MNN::Timer _t;
    // dit
    auto generated_mel = ditForward(real_size, codec_ptr, noise_ptr);
    generated_mel = _Slice(generated_mel, _var<int>({0, 0, dit_left_padding * 2}, {3}), _var<int>({-1, -1, mel_size}, {3}));
    mMelBuffer = (mMelBuffer == nullptr) ? generated_mel : _Concat({mMelBuffer, generated_mel}, -1);
    dit_left_padding = dit_left_context;
    dit_start_index += (chunk_size - dit_left_padding - dit_right_padding);
    // bigvga
    auto generated_waveform = bigvganForward(mMelBuffer);
    // append waveform to mWaveformBuffer
    auto ptr = generated_waveform->readMap<float>() + vocoder_left_pad * vocoder_upsample_rate;
    auto size = generated_waveform->getInfo()->size - (vocoder_left_pad + vocoder_right_pad) * vocoder_upsample_rate;
    mWaveformBuffer.insert(mWaveformBuffer.end(), ptr, ptr + size);
    vocoder_left_pad = vocoder_left_context;
    mMelBuffer = _Slice(mMelBuffer, _var<int>({0, 0, -vocoder_left_pad - vocoder_right_pad}, {3}), _var<int>({-1, -1, -1}, {3}));
    mContext->audio_us += _t.durationInUs();
    if (mWavformCallback) {
        bool res = mWavformCallback(ptr, size, last_chunk);
        if (!res) { return; }
    }
    if (talker_done && !last_chunk) {
        token2wav(true);
    }
}

VARP Talker::token2wav(const std::vector<int>& codec_tokens) {
    auto generated_mel = ditForward(codec_tokens.size(), codec_tokens.data());
    auto waveform = bigvganForward(generated_mel);
    return waveform;
}

int Talker::sample(Express::VARP logits, int offset, int size) {
    int token = Llm::sample(logits, offset, size);
    if (mStreamWithDecode) {
        token2wav();
    }
    return token;
}

void Talker::generate() {
    if (!doGenerate()) { return; }
    mTalkerEmbeds.push_back(mTextEos);
    auto input_embeds = _Concat({mTalkerEmbeds[0], mTextBos + mCodecPad, mTalkerEmbeds[1] + mCodecBos}, 1);
    // push 2 token ids
    mPositionIds.push_back();
    mPositionIds.push_back();
    mContext->prompt_len = input_embeds->getInfo()->dim[1];
    MNN::Timer _t;
    auto logits = forward(input_embeds);
    mContext->current_token = sample(logits);
    mContext->history_tokens.push_back(mContext->current_token);
    mContext->output_tokens.push_back(mContext->current_token);
    mContext->prefill_us += _t.durationInUs();
    _t.reset();
    for (int i = 1; i < mMaxNewTokens; i++) {
        input_embeds = embedding({mContext->current_token});
        if (i + 1 < mTalkerEmbeds.size()) {
            input_embeds = input_embeds + mTalkerEmbeds[i + 1];
        } else {
            mTalkerEmbeds.clear();
            input_embeds = input_embeds + mTextPad;
        }
        auto logits = forward(input_embeds);
        mContext->current_token = sample(logits);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);

        if (mContext->current_token == 8292 || mContext->current_token == 8294) {
            break;
        }
    }
    mContext->decode_us += _t.durationInUs();
    token2wav(true);
}

void Talker::setPostionIds(const MropeInfo& positionIds) {
    if (!doGenerate()) { return; }
    mPositionIds = MropeInfo(positionIds);
}

void Talker::addTalkerEmbeds(VARP talker_embeds) {
    if (!doGenerate()) { return; }
    mTalkerEmbeds.push_back(_Clone(talker_embeds, true));
}

} // namespace Transformer
} // namespace MNN
