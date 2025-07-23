//
//  generate.hpp
//
//  Created by MNN on 2025/06/09.
//

#include "generate.hpp"
#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"
#include "lookahead.hpp"

using namespace MNN::Express;

namespace MNN {
namespace Transformer {

std::shared_ptr<Generation> GenerationStrategyFactory::create(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config, bool canSpec) {
    std::shared_ptr<Generation> res;
    if(canSpec) {
        if(config->speculative_type() == "lookahead") {
            res.reset(new LookaheadGeneration(llm, context, config));
        } else if(config->speculative_type() == "mtp") {
            res.reset(new MtpGeneration(llm, context, config));
        } else {
            // autoregressive generation
            res.reset(new ArGeneration(llm, context, config));
        }
    } else {
        // autoregressive generation
        res.reset(new ArGeneration(llm, context, config));
    }
    return res;
}

ArGeneration::ArGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}
void ArGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;
    int len = 0;
    while (len < max_token) {
        AUTOTIME;
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
        // Update gen seq
        mLlm->updateContext(0, 1);
        MNN::Timer _t;
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // mContext->history_tokens.push_back(mContext->current_token);
        mLlm->mMeta->remove = 0;
        auto outputs = mLlm->forwardVec({mContext->current_token});
        if(outputs.empty()) {
            break;
        }
        auto logits = outputs[0];
        // Update all seq
        mLlm->updateContext(1, 0);
        len++;
        if (nullptr == logits.get()) {
            break;
        }
        if (logits->getInfo()->size == 0) {
            break;
        }
        mContext->current_token = mLlm->sample(logits);
        mContext->decode_us += _t.durationInUs();
        if (mLlm->is_stop(mContext->current_token)) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }
    }
}

int Generation::draftVerify(VARP logits, const std::vector<int> &drafts, bool& stop) {
    // verify draft token whether be accepted
    int i_dft = 1;
    {
        //AUTOTIME;
        for(; i_dft < drafts.size(); i_dft++) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size - (drafts.size() - i_dft + 1) * sample_size;
            
            auto predict = mLlm->sample(logits, sample_offset, sample_size);
            
            // stop token just break the process
            if (mLlm->is_stop(predict) && nullptr != mContext->os) {
                mContext->current_token = predict;
                *mContext->os << mContext->end_with << std::flush;
                stop = true;
                break;
            }
            // draft token id not match
            if(predict != drafts[i_dft]) {
                mContext->current_token = predict;
                break;
            }

            if (nullptr != mContext->os) {
                *mContext->os << mLlm->tokenizer_decode(predict);
                *mContext->os << std::flush;
            }
        }
        // all drafts are corrcet!
        if(i_dft == drafts.size()) {
            auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
            auto sample_offset = logits->getInfo()->size -  sample_size;
            
            auto predict = mLlm->sample(logits, sample_offset, sample_size);
            mContext->current_token = predict;
        }
    }

    return i_dft;
}



} // namespace Transformer
} // namespace MNN
