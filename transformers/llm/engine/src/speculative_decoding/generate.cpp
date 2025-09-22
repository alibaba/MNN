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
        // Update gen seq
        mContext->current_token = mLlm->sample(param.outputs[0], param.validLogitStart, param.validLogitSize);
        mContext->history_tokens.push_back(mContext->current_token);
        mContext->output_tokens.push_back(mContext->current_token);
        mLlm->updateContext(0, 1);
        if (mLlm->is_stop(mContext->current_token)) {
            if (nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
            }
            break;
        }
        // Decode and Output
        MNN::Timer _t;
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        
        // Compute Next Logits
        auto outputs = mLlm->forwardVec({mContext->current_token});
        if(outputs.empty()) {
            break;
        }
        // Update input seq
        mLlm->updateContext(1, 0);
        mContext->decode_us += _t.durationInUs();
        len++;
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
            if (mLlm->is_stop(predict)) {
                mContext->current_token = predict;
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
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
