//
//  omni.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OMNI_hpp
#define OMNI_hpp

#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

namespace MNN {
using namespace Express;
namespace Transformer {

static const char* const kOmniMultimodalRegex = "<(img|audio|video)>(.*?)</\\1>";

class MropeInfo {
public:
    int back() const {
        if (mW.empty()) {
            return 0;
        }
        return mW.back();
    }
    int currentIdx() const {
        if (mW.empty()) {
            return 0;
        }
        return std::max(std::max(mT.back(), mH.back()), std::max(mW.back(), mX.back())) + 1;
    }
    void push_back(int t, int h, int w) { push_back(t, h, w, w); }
    void push_back(int t, int h, int w, int x) {
        mT.push_back(t);
        mH.push_back(h);
        mW.push_back(w);
        mX.push_back(x);
    }
    void push_back(int t) { push_back(t, t, t); }
    void push_back() {
        int cur_idx = currentIdx();
        push_back(cur_idx, cur_idx, cur_idx);
    }
    void prependTextPositions(size_t count) {
        if (count == 0) {
            return;
        }
        const int offset = static_cast<int>(count);
        std::vector<int> prefix(count);
        for (size_t i = 0; i < count; ++i) {
            prefix[i] = i;
        }
        for (auto* positions : {&mT, &mH, &mW, &mX}) {
            for (auto& position : *positions) {
                position += offset;
            }
            positions->insert(positions->begin(), prefix.begin(), prefix.end());
        }
    }
    void clear() {
        mT.clear();
        mH.clear();
        mW.clear();
        mX.clear();
    }
    std::vector<int> mT, mH, mW, mX;
};

class Talker : public Llm {
public:
    Talker(std::shared_ptr<LlmConfig> config) : Talker(config, nullptr) {}
    Talker(std::shared_ptr<LlmConfig> config, Llm* thinker) : Llm(config), mThinker(thinker) {}
    ~Talker();
    bool load() override;
    void setProcessorRuntimeManager(std::shared_ptr<Executor::RuntimeManager> processorRuntimeManager);
    void generate_init(std::ostream* os = nullptr, const char* end_with = nullptr) override;
    Express::VARP embedding(const std::vector<int>& input_ids) override;
    Express::VARP gen_position_ids(int seq_len) override;
    int sample(Express::VARP logits, int offset = 0, int size = 0) override;
    void setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) override;
    void generate();
    bool generateQwen3TTS(const std::string& prompt, int maxFrames, const std::string& refAudio);
    void stepPrefill();
    void stepForward(int stepIdx);
    void finalize();
    void setPostionIds(const MropeInfo& positionIds);
    void addTalkerEmbeds(VARP talker_embeds);
    bool hasEmbeds() const { return !mTalkerEmbeds.empty(); }
    bool doGenerate() const { return mWavformCallback != nullptr; }
    int maxNewTokens() const { return mMaxNewTokens; }
    bool mStreamWithDecode = false;
    bool mInterleaved = false;

private:
    struct WavChunk {
        std::vector<int> codec_tokens;
        std::vector<float> noise;
        int mel_slice_start = 0;
        int mel_slice_size = -1;
        bool is_last = false;
        std::vector<float> mel;
        std::vector<int> mel_dims;
    };

    int mMaxNewTokens = 2048, mTextBosToken = 151872, mTextEosToken = 151861, mTextPadToken = 151859,
        mCodecBosToken = 8293, mCodecPadToken = 8292;
    VARP mTextBos, mTextEos, mTextPad, mCodecBos, mCodecPad, mSpk, mCond;
    MropeInfo mPositionIds;
    std::vector<VARP> mTalkerEmbeds;
    std::shared_ptr<Module> mPreDit, mDit, mBigvgan;
    std::shared_ptr<Module> mQwen3PromptEmbedder, mQwen3AudioCodePredictor, mQwen3CodecEmbedder, mQwen3SpeechDecoder;
    std::shared_ptr<Module> mQwen3SpeakerEncoder;
    std::shared_ptr<DiskEmbedding> mQwen3TextEmbedding, mQwen3AudioCodePredictorEmbedding;
    Llm* mThinker;
    std::shared_ptr<Executor::RuntimeManager> mProcessorRuntimeManager, mQwen3RuntimeManager;
    std::shared_ptr<Executor::RuntimeManager> mQwen3CpRuntimeManager;
    std::shared_ptr<KVMeta> mQwen3CpMeta;
    VARP ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise = nullptr);
    VARP bigvganForward(VARP mel);
    VARP token2wav(const std::vector<int>& codec_tokens);
    void token2wav(bool talker_done = false);

    std::vector<float> mInitialNoise, mWaveformBuffer;
    VARP mMelBuffer = nullptr;
    const int dit_chunk_size = 60, dit_left_context = 24, dit_right_context = 12, dit_right_padding = dit_right_context,
              vocoder_left_context = 8, vocoder_right_context = 8, vocoder_right_pad = vocoder_right_context,
              vocoder_upsample_rate = 240;
    int dit_left_padding = 0, dit_start_index = 0, vocoder_left_pad = 0;
    std::function<bool(const float*, size_t, bool)> mWavformCallback = nullptr;
    // async token2wav pipeline (true parallelism via cloned modules)
    void startAsyncWorker();
    void stopAsyncWorker();
    void ditWorkerLoop();
    void vocoderWorkerLoop();
    void processWavChunk(WavChunk& chunk);
    void trySubmitChunkAsync(bool talker_done);
    VARP ditForwardAsync(const int codec_size, const int* codec_tokens, const float* initial_noise);
    std::thread mDitWorkerThread;
    std::thread mVocoderWorkerThread;
    std::mutex mWavQueueMutex;
    std::condition_variable mWavQueueCond;
    std::queue<WavChunk> mWavQueue;
    std::mutex mMelQueueMutex;
    std::condition_variable mMelQueueCond;
    std::queue<WavChunk> mMelQueue;
    std::atomic<bool> mWavWorkerRunning{false};
    std::atomic<bool> mWavLastDone{false};
    bool mAsyncToken2Wav = false;
    // cloned modules for worker thread — independent Session/Runtime, shared weights
    std::shared_ptr<Module> mPreDit_async, mDit_async, mBigvgan_async;
    VARP mSpk_async, mCond_async;
};

class Omni : public Embedding {
public:
    Omni(std::shared_ptr<LlmConfig> config);
    ~Omni();
    bool load() override;
    std::vector<Express::VARP> forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos,
                                          Express::VARPS extraArgs = {}) override;
    std::vector<int> tokenizer_encode(const std::string& query) override;
    std::vector<int> tokenizer_encode(const MultimodalPrompt& multimodal_input) override;
    Express::VARP embedding(const std::vector<int>& input_ids) override;
    Express::VARP gen_attention_mask(int seq_len) override;
    Express::VARP gen_position_ids(int seq_len) override;
    bool prepareHistoryReplay(const std::vector<int>& input_ids) override;
    void response(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr,
                  int max_new_tokens = -1) override;
    void setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) override;
    void generateWavform() override;
    bool generateTTS(const std::string& text, const std::string& language = "english", int max_new_tokens = -1,
                     const std::string& ref_audio = "") override;
    // Embedding API — single ExecutorScope wrapping the forward pass (same pattern as forwardVec)
    Express::VARP ids_embedding(const std::vector<int>& ids) override;
    std::vector<int> visionProcess(VARP image);

private:
    struct VisionProcessor;
    struct AudioProcessor;
    bool initProcessorRuntime();
    VARPS runEncoder(Module* module, const VARPS& inputs, const ujson::json& processor, bool preserve = false);
    std::vector<int> processMediaFile(const std::string& mode, std::string info);
    std::vector<int> processMediaContent(const std::string& mode, const std::string& content,
                                         const MultimodalPrompt& prompt);
    void addPositionIds(int count);
    void responseInterleaved(const std::vector<int>& input_ids, std::ostream* os, const char* end_with,
                             int max_new_tokens);

    void setImageSize(int height, int width);
    std::vector<int> visionProcess(const std::string& file);
    std::vector<int> videoProcess(const std::string& file);
    std::vector<int> videoProcess(const PromptVideoPart& video);
    std::shared_ptr<Module> mVisionModule;
    ujson::json mVisionConfig;
    std::unique_ptr<VisionProcessor> mVisionProcessor;
    Dimensionformat mVisionInputOrder = NCHW;
    int mVisionHeight = 448, mVisionWidth = 448, mVisionStart = 151857, mVisionEnd = 151858, mVisionPad = 151859;
    int mVideoPad = -1, mVisionGlobal = 49152, mVisionSizeUnit = 1, mVisionMaxSize = 2048;
    float mVideoFps = 2.0f, mVideoPositionIdPerSecond = 0.0f;
    int mVideoMaxFrames = 768, mVideoMaxPixels = 768 * 28 * 28, mVideoMaxVisionTokens = 4096;
    int mVisionNum = 0;
    bool mVisionSizeOverridden = false;
    std::vector<float> mVisionMean, mVisionNorm;

    std::vector<int> audioProcess(const std::string& file);
    std::vector<int> audioProcess(VARP waveform);
    std::shared_ptr<Module> mAudioModule;
    ujson::json mAudioConfig;
    std::unique_ptr<AudioProcessor> mAudioProcessor;
    int mAudioPad = 151646;

    std::vector<VARP> mExtraArgs, mVisionEmbeddings, mAudioEmbeddings, mDeepStackEmbeddings;
    MropeInfo mPositionIds;
    std::shared_ptr<Talker> mTalker;
    int64_t mThinkerElapsedUs = 0;
    bool mIsEmbedding = false;
};
} // namespace Transformer
} // namespace MNN
#endif // OMNI_hpp
