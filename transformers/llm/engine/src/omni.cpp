//
//  omni.cpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <regex>
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
    }
    if (config->is_audio()) {}
}

void Omni::load() {
    Llm::load();
    if (mConfig->has_talker()) {
        mTalker.reset(new Talker(mConfig, this));
        mTalker->load();
    }
    if (mConfig->mllm_config_.empty()) {
        mProcessorRuntimeManager = mRuntimeManager;
    } else {
        ScheduleConfig config;
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
        mProcessorRuntimeManager->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
        mProcessorRuntimeManager->setHint(MNN::Interpreter::MEM_ALLOCATOR_TYPE, 0);
        mProcessorRuntimeManager->setHint(MNN::Interpreter::QKV_QUANT_OPTIONS, mConfig->quant_qkv());
        mProcessorRuntimeManager->setHint(MNN::Interpreter::KVCACHE_SIZE_LIMIT, mConfig->kvcache_limit());
        std::string tmpPath = mConfig->tmp_path();
        if (mConfig->kvcache_mmap()) {
            mProcessorRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_PATH_KVCACHE_DIR);
        }
        if (mConfig->use_mmap()) {
            mProcessorRuntimeManager->setExternalPath(tmpPath, MNN::Interpreter::EXTERNAL_WEIGHT_DIR);
        }
    }
    Module::Config module_config;
    module_config.shapeMutable = true;
    module_config.rearrange    = true;
    if (mConfig->is_visual()) {
        mVisionModule.reset(Module::load({}, {}, mConfig->visual_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
    if (mConfig->is_audio()) {
        mAudioModule.reset(Module::load({}, {}, mConfig->audio_model().c_str(), mProcessorRuntimeManager, &module_config));
    }
}

std::vector<int> Omni::visionProcess(const std::string& file) {
#ifdef LLM_SUPPORT_VISION
    VARP image = MNN::CV::imread(file);
    if (image == nullptr) {
        MNN_PRINT("Omni Can't open image: %s\n", file.c_str());
        return std::vector<int>(0);
    }
    Timer _t;
    VARP image_embedding;

    if (mVisionModule->getInfo()->inputNames[0] == "patches") {
        bool hasWindowIndex = mVisionModule->getInfo()->inputNames.size() == 4 &&
                              mVisionModule->getInfo()->inputNames[3] == "window_index";
        // Qwen2-VL / Qwen2.5-VL
        mVisionHeight = round(mVisionHeight / 28.0) * 28;
        mVisionWidth = round(mVisionWidth / 28.0) * 28;
        image        = MNN::CV::resize(image, {mVisionHeight, mVisionWidth}, 0, 0,
                                     MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                     mVisionMean, mVisionNorm);
        image        = Express::_Unsqueeze(image, {0});
        image        = Express::_Convert(image, NCHW);
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
        image_embedding = mVisionModule->onForward(moduleInputs)[0];
#ifdef DEBUG_IMAGE
        image_embedding->setName("image_embeds");
        MNN::Express::Variable::save({image_embedding}, "output.mnn");
#endif
    } else {
        mVisionHeight = UP_DIV(mVisionHeight, mVisionSizeUnit) * mVisionSizeUnit;
        mVisionWidth = UP_DIV(mVisionWidth, mVisionSizeUnit) * mVisionSizeUnit;
        image           = MNN::CV::resize(image, {mVisionHeight, mVisionWidth}, 0, 0,
                                          MNN::CV::INTER_LINEAR, MNN::CV::COLOR_BGR2RGB,
                                          mVisionMean, mVisionNorm);
        image           = Express::_Unsqueeze(image, {0});
        image           = Express::_Convert(image, NC4HW4);
        image_embedding = mVisionModule->forward(image);
    }
    mContext->vision_us = _t.durationInUs();
    mVisionEmbeddings.push_back(image_embedding);
    int visual_len = image_embedding->getInfo()->dim[0];
    std::vector<int> img_ids(visual_len, mVisionPad);
    img_ids.insert(img_ids.begin(), mVisionStart);
    img_ids.push_back(mVisionEnd);
    return img_ids;
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
    // int sample_rate      = load_res.second;
    int wav_len          = waveform->getInfo()->dim[0];
    int hop_length       = 160;
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

std::vector<int> Omni::tokenizer_encode(const std::string& prompt) {
    // split query
    std::regex multimode_regex("<(img|audio)>(.*?)</\\1>");
    std::string::const_iterator searchStart(prompt.cbegin());
    std::smatch match;
    std::vector<std::string> img_infos;
    std::vector<int> ids{};

    mPositionIds.clear();
    while (std::regex_search(searchStart, prompt.cend(), match, multimode_regex)) {
        // std::cout << "img match: " << match[1].str() << std::endl;
        auto txt_ids = mTokenizer->encode(match.prefix().str());
        addPositionIds(txt_ids.size());
        ids.insert(ids.end(), txt_ids.begin(), txt_ids.end());
        auto mul_ids = multimodeProcess(match[1].str(), match[2].str());
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

VARP Omni::embedding(const std::vector<int>& input_ids) {
    if (input_ids.size() == 1) {
        return Llm::embedding(input_ids);
    }
    std::vector<VARP> embeddings;
    std::vector<int> position_ids;
    int vision_idx = 0, audio_idx = 0;
    std::vector<int> cur_txt_ids;
    bool in_audio = false;
    for (int i = 0; i < input_ids.size(); i++) {
        int id = input_ids[i];
        // audio
        if (in_audio) {
            if (id == mAudioPad) {
                continue;
            } else {
                cur_txt_ids.clear();
                in_audio = false;
            }
        } else if (id == mAudioPad) {
            auto txt_embedding = Llm::embedding(cur_txt_ids);
            auto mul_embedding = mAudioEmbeddings[audio_idx++];
            embeddings.push_back(txt_embedding);
            embeddings.push_back(mul_embedding);
            in_audio = true;
        }
        // vision
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
        if (mTalker) {
            mTalker->setPostionIds(mPositionIds);
        }
    }
    return positionIds;
}

Express::VARP Omni::forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos) {
    VARP logits;
    auto logitsIndex = _var<int>({-1}, {1});
    if (mConfig->all_logits()) {
        logitsIndex = _var<int>({0}, {1});
    }
    std::vector<Express::VARP> outputs;
    outputs = mCurrentModules.back()->onForward({hiddenState, mask, inputPos, logitsIndex});
    if (outputs.empty()) {
        return nullptr;
    }
    if (mTalker && outputs.size() > 1) {
        mTalker->addTalkerEmbeds(outputs[1]);
    }
    logits = outputs[0];
    mMeta->sync();
    return logits;
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
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    initRuntime();
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
    mModules[0].reset(Module::load({"inputs_embeds", "attention_mask", "position_ids"},
                                    {"logits"}, mConfig->talker_model().c_str(), mRuntimeManager, &module_config));
    // dit
    mPreDit.reset(Module::load({"cond", "spk", "code"}, {"code_embeds", "rope", "mask"},
                                mConfig->predit_model().c_str(), mRuntimeManager, &module_config));
    mDit.reset(Module::load({"x", "code_embeds", "rope", "mask", "time"}, {"mel"},
                            mConfig->dit_model().c_str(), mRuntimeManager, &module_config));
    // bigvgan
    mBigvgan.reset(Module::load({"generated_mel"},
                                {"waveform"}, mConfig->bigvgan_model().c_str(), mRuntimeManager, &module_config));
    mDecodeModules.resize(mModules.size());
    mDecodeModules[0].reset(Module::clone(mModules[0].get()));
    mPrefillModules = mModules;
}

void Talker::generate_init(std::ostream* os, const char* end_with) {
    if (!doGenerate()) { return; }
    {
        mContext->os = os;
        if (nullptr != end_with) {
            mContext->end_with = end_with;
        }
        if (!mContext->generate_str.empty()) {
            mContext->generate_str.clear();
        }
        mContext->gen_seq_len = 0;
        mContext->prefill_us  = 0;
        mContext->decode_us   = 0;
        mContext->current_token = 0;
        mContext->all_seq_len = 0;
        mContext->history_tokens.clear();
        mMeta->remove = mMeta->previous;
        mContext->output_tokens.clear();
        mCurrentModules = mPrefillModules;
    }
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

VARP Talker::forward(VARP input_embeds) {
    auto input_shape = input_embeds->getInfo()->dim;
    int seq_len = input_shape[1];
    mMeta->add = seq_len;
    auto attention_mask = gen_attention_mask(seq_len);
    auto position_ids = gen_position_ids(seq_len);
    auto outputs = mCurrentModules.back()->onForward({input_embeds, attention_mask, position_ids});
    mContext->all_seq_len += seq_len;
    mContext->gen_seq_len++;
    mMeta->sync();
    return outputs[0];
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
    int token = sample(logits);
    mContext->prefill_us += _t.durationInUs();
    _t.reset();
    for (int i = 1; i < mMaxNewTokens; i++) {
        input_embeds = embedding({token});
        if (i + 1 < mTalkerEmbeds.size()) {
            input_embeds = input_embeds + mTalkerEmbeds[i + 1];
        } else {
            mTalkerEmbeds.clear();
            input_embeds = input_embeds + mTextPad;
        }
        auto logits = forward(input_embeds);
        token = sample(logits);
        if (token == 8292 || token == 8294) {
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
