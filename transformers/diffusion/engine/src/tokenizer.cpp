#include "tokenizer.hpp"

#include <algorithm>

#include "core/Macro.h"

#if defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
#include "../../../llm/engine/src/tokenizer/tokenizer.hpp"
#endif

namespace MNN {
namespace DIFFUSION {

static std::string BuildMtokPath(const std::string& filePath) {
    return filePath + "/tokenizer.mtok";
}

MtokTokenizer::MtokTokenizer(Style style, int bosId, int eosId) : mStyle(style), mBosId(bosId), mEosId(eosId) {}

MtokTokenizer::~MtokTokenizer() {
#if defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
    delete mTokenizer;
    mTokenizer = nullptr;
#endif
}

bool MtokTokenizer::load(const std::string& filePath) {
#if defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
    auto mtokPath = BuildMtokPath(filePath);
    delete mTokenizer;
    mTokenizer = MNN::Transformer::Tokenizer::createTokenizer(mtokPath);
    if (mTokenizer == nullptr) {
        MNN_ERROR("Failed to create tokenizer from %s\n", mtokPath.c_str());
    }
    return mTokenizer != nullptr;
#else
    (void)filePath;
    MNN_ERROR("Diffusion tokenizer requires MNN_BUILD_LLM=ON and tokenizer.mtok\n");
    return false;
#endif
}

std::vector<int> MtokTokenizer::encodeSingle(const std::string& sentence, int maxlen) const {
#if defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
    if (mTokenizer == nullptr) {
        return {};
    }
    std::vector<int> ids = mTokenizer->encode(sentence);
    if (mBosId >= 0 && (ids.empty() || ids.front() != mBosId)) {
        ids.insert(ids.begin(), mBosId);
    }
    if (mEosId >= 0 && (ids.empty() || ids.back() != mEosId)) {
        ids.push_back(mEosId);
    }
    if (maxlen > 0) {
        if ((int)ids.size() > maxlen) {
            ids.resize(maxlen);
            if (mEosId >= 0) {
                ids[maxlen - 1] = mEosId;
            }
        } else {
            while ((int)ids.size() < maxlen) {
                ids.push_back(0);
            }
        }
    }
    return ids;
#else
    (void)sentence;
    (void)maxlen;
    return {};
#endif
}

std::vector<int> MtokTokenizer::encode(const std::string& sentence, int maxlen) {
#if defined(MNN_DIFFUSION_WITH_LLM_TOKENIZER)
    if (mStyle == Style::kPair) {
        std::vector<int> ids(maxlen * 2, 0);
        auto uncond = encodeSingle("", maxlen);
        auto cond = encodeSingle(sentence, maxlen);
        std::copy(uncond.begin(), uncond.end(), ids.begin());
        std::copy(cond.begin(), cond.end(), ids.begin() + maxlen);
        return ids;
    }
    return encodeSingle(sentence, maxlen);
#else
    (void)sentence;
    (void)maxlen;
    return {};
#endif
}

} // namespace DIFFUSION
} // namespace MNN
