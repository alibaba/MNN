//
//  omni.hpp
//
//  Created by MNN on 2025/04/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OMNI_hpp
#define OMNI_hpp

#include "llm/llm.hpp"

namespace MNN {
using namespace Express;
namespace Transformer {

class MropeInfo {
public:
    MropeInfo() {}
    MropeInfo(const MropeInfo& info) {
        mT = info.mT;
        mH = info.mH;
        mW = info.mW;
    }
    int back() {
        if (mW.empty()) {
            return 0;
        }
        return mW.back();
    }
    int currentIdx() {
        if (mW.empty()) {
            return 0;
        }
        return back() + 1;
    }
    void push_back(int t, int h, int w) {
        mT.push_back(t);
        mH.push_back(h);
        mW.push_back(w);
    }
    void push_back(int t) {
        push_back(t, t, t);
    }
    void push_back() {
        int cur_idx = currentIdx();
        push_back(cur_idx, cur_idx, cur_idx);
    }
    void clear() {
        mT.clear();
        mH.clear();
        mW.clear();
    }
    std::vector<int> mT, mH, mW;
};

class Talker : public Llm {
public:
    Talker(std::shared_ptr<LlmConfig> config) : Llm(config), mThinker(nullptr) {}
    Talker(std::shared_ptr<LlmConfig> config, Llm* thinker) : Llm(config), mThinker(thinker) {}
    ~Talker() {}
    virtual bool load() override;
    virtual void generate_init(std::ostream* os = nullptr, const char* end_with = nullptr) override;
    virtual Express::VARP embedding(const std::vector<int>& input_ids) override;
    virtual Express::VARP gen_position_ids(int seq_len) override;
    virtual int sample(Express::VARP logits, int offset = 0, int size = 0) override;
    virtual void setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) override;
    VARP ditForward(const int codec_size, const int* codec_tokens, const float* initial_noise = nullptr);
    VARP bigvganForward(VARP mel);
    VARP token2wav(const std::vector<int>& codec_tokens);
    void token2wav(bool talker_done = false);
    void generate();
    void setPostionIds(const MropeInfo& positionIds);
    void addTalkerEmbeds(VARP talker_embeds);
    // is generate
    bool doGenerate() { return mWavformCallback != nullptr; }
    // is decode with token2wav
    bool mStreamWithDecode = false;
private:
    int mMaxNewTokens = 2048, mTextBosToken = 151872, mTextEosToken = 151861,
        mTextPadToken = 151859, mCodecBosToken = 8293, mCodecPadToken = 8292;
    VARP mTextBos, mTextEos, mTextPad, mCodecBos, mCodecPad, mSpk, mCond;
    MropeInfo mPositionIds;
    std::vector<VARP> mTalkerEmbeds;
    std::shared_ptr<Module> mPreDit, mDit, mBigvgan;
    Llm* mThinker;
    // stream generate
    std::vector<float> mInitialNoise, mWaveformBuffer;
    VARP mMelBuffer = nullptr;
    const int dit_chunk_size = 60, dit_left_context = 24,
        dit_right_context = 12, dit_right_padding = dit_right_context,
        vocoder_left_context = 8, vocoder_right_context = 8,
        vocoder_right_pad = vocoder_right_context, vocoder_upsample_rate = 240;
    int dit_left_padding = 0, dit_start_index = 0, vocoder_left_pad = 0;
    std::function<bool(const float*, size_t, bool)> mWavformCallback = nullptr;
};

class Omni : public Llm {
public:
    Omni(std::shared_ptr<LlmConfig> config);
    ~Omni() {
        mVisionModule.reset();
        mAudioModule.reset();
    }
    virtual bool load() override;
    virtual std::vector<Express::VARP> forwardRaw(Express::VARP hiddenState, Express::VARP mask, Express::VARP inputPos, Express::VARPS extraArgs) override;
    virtual std::vector<int> tokenizer_encode(const std::string& query) override;
    virtual std::vector<int> tokenizer_encode(const MultimodalPrompt& multimodal_input) override;
    virtual Express::VARP embedding(const std::vector<int>& input_ids) override;
    virtual Express::VARP gen_position_ids(int seq_len) override;
    virtual void response(const std::vector<int>& input_ids, std::ostream* os = &std::cout, const char* end_with = nullptr, int max_new_tokens = -1) override;
    virtual void setWavformCallback(std::function<bool(const float*, size_t, bool)> callback) override;
    virtual void generateWavform() override;
    // some models preprocess function
    std::vector<int> visionProcess(VARP image);
    std::vector<int> defaultVisionProcess(VARP image);
    std::vector<int> qwen2VisionProcess(VARP image);
    std::vector<int> smolvlmVisionProcess(VARP image);
    std::vector<int> minicpmVisionProcess(VARP image);
private:
    int mVisionHeight = 448, mVisionWidth = 448, mVisionStart = 151857,
        mVisionEnd = 151858, mVisionPad = 151859, mAudioPad = 151646;
    int mVisionGlobal = 49152;
    int mVisionSizeUnit = 1, mVisionMaxSize = 2048;
    int mVisionNum = 0;
    std::vector<float> mVisionMean{122.7709383, 116.7460125, 104.09373615};
    std::vector<float> mVisionNorm{0.01459843, 0.01500777, 0.01422007};
    std::vector<int> multimodeProcess(const std::string& mode, std::string info);
    std::vector<int> visionProcess(const std::string& file);
    std::vector<int> audioProcess(const std::string& file);
    std::vector<int> audioProcess(MNN::Express::VARP waveform);
    std::vector<int> processImageContent(const std::string& content, const std::map<std::string, PromptImagePart>& images);
    std::vector<int> processAudioContent(const std::string& content, const std::map<std::string, PromptAudioPart>& audios);
    std::shared_ptr<Module> mVisionModule, mAudioModule;
    std::vector<VARP> mExtraArgs, mVisionEmbeddings, mAudioEmbeddings, mDeepStackEmbeddings;
    std::shared_ptr<Talker> mTalker;
    // m_rope position ids
    void addPositionIds(int t, int h = -1, int w = -1);
    MropeInfo mPositionIds;
};

}
}
#endif // OMNI_hpp
