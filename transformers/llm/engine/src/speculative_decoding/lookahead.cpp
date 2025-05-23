//
//  lookahead.cpp
//
//  Created by MNN on 2025/04/09.
//

#include <MNN/AutoTime.hpp>
#include "llm/llm.hpp"
#include "ngram.hpp"
#include "../llmconfig.hpp"
#include "../kvmeta.hpp"
#include "lookahead.hpp"
//#define DUMP_PROFILE_INFO

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

void Llm::speculativeGenerate(int max_token) {
    // remove first prefill token, for draft match
    mContext->history_tokens.pop_back();
    mContext->output_tokens.pop_back();

    int len = 0;
    ngram_cache<ngram_value> prompt_ngram_cache;
    ngram_cache<ngram_ordered_value> prompt_ngram_ordered_cache;

    if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
        ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, mContext->history_tokens.size());
    } else {
        ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, mContext->history_tokens.size());
    }
    
    // user provided info to create ngram
    {
        auto prior_prompt_file = mConfig->lookup_file();
        std::ifstream file(prior_prompt_file);

        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string user_set_prompt = buffer.str();
            file.close();
            std::vector<int> user_ids = tokenizer_encode(user_set_prompt);
            
            if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            } else {
                ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            }
        }
    }
    // speculative total decode token numbers
    int spl_decode = 0;
    // speculative accept token numbers
    int spl_accept = 0;
    // autoregression number of times
    int arg_count = 0;
    // speculative number of times
    int spl_count = 0;

    while (len < max_token) {
        MNN::Timer _t;
        std::vector<int> drafts;
        drafts.push_back(mContext->current_token);
        
        auto decodeStr = tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        
        {
            // draft is "," or "." or ":" or "、", match maybe confusion
            bool confuse = decodeStr == "," || decodeStr == "." || decodeStr == ":" || \
                decodeStr == "、" || decodeStr == ";" || \
                decodeStr == "，" || decodeStr == "。" || decodeStr == "：" || \
                // Chinese comma and semicolon
                decodeStr == "、" || decodeStr == "；";
            MatchStrictLevel level = mStrictLevel;
            // for confuse key, set match_strictness to high
            if(confuse) {
                level = MatchStrictLevel::HIGH_LEVEL;
            }
            // generate draft tokens
            if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                ngram_cache_search(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts, mDraftLength, level);
            } else {
                ngram_cache_search(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts, mDraftLength, level);
            }
            mMeta->add = drafts.size();

            AUTOTIME;
            // do draft token parallel verify
            auto logits = forward(drafts);
            if (nullptr == logits.get()) {
                break;
            }
            if (logits->getInfo()->size == 0) {
                break;
            }

            // verify draft token whether be accepted
            int i_dft = 1;
            {
                //AUTOTIME;
                for(; i_dft < drafts.size(); i_dft++) {
                    auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
                    auto sample_offset = logits->getInfo()->size - (drafts.size() - i_dft + 1) * sample_size;
                    
                    auto predict = sample(logits, sample_offset, sample_size);
                    
                    // stop token just break the process
                    if (is_stop(predict) && nullptr != mContext->os) {
                        mContext->current_token = predict;
                        *mContext->os << mContext->end_with << std::flush;
                        break;
                    }
                    // draft token id not match
                    if(predict != drafts[i_dft]) {
                        mContext->current_token = predict;
                        break;
                    }
                    
                    mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin() + i_dft, drafts.begin() + i_dft + 1);
                    mContext->output_tokens.push_back(drafts[i_dft]);

                    if (nullptr != mContext->os) {
                        *mContext->os << tokenizer_decode(predict);
                        *mContext->os << std::flush;
                    }
                }
                // all drafts are corrcet!
                if(i_dft == drafts.size()) {
                    auto sample_size = logits->getInfo()->dim[logits->getInfo()->dim.size() - 1];
                    auto sample_offset = logits->getInfo()->size -  sample_size;
                    
                    auto predict = sample(logits, sample_offset, sample_size);
                    mContext->current_token = predict;
                }
            }

            // update ngram for each decoding
            if(mUpdateNgram) {
                if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                    ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                } else {
                    ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                }
            }

            // clear dirty kv-cache
            mMeta->remove = drafts.size() - i_dft;
            len += i_dft;
            
            // update context state
            mContext->all_seq_len -= mMeta->remove;
            mContext->gen_seq_len += (i_dft - 1);
            
        #ifdef DUMP_PROFILE_INFO
            MNN_PRINT("\ndraft num:%d, adopt num:%d\n", drafts.size(), i_dft);
            if(drafts.size() > 1) {
                spl_decode += drafts.size();
                spl_accept += i_dft;
                spl_count++;
            } else {
                arg_count++;
            }
        #endif
            if (is_stop(mContext->current_token) && nullptr != mContext->os) {
                *mContext->os << mContext->end_with << std::flush;
                break;
            }
        }
        mContext->decode_us += _t.durationInUs();
    }
#ifdef DUMP_PROFILE_INFO
    // adopt speculative decoding rate
    float spl_rate = 100.0 * spl_count / (spl_count + arg_count);
    // draft accept rate if adopt speculative decoding
    float spl_accept_rate = 100.0 * spl_accept / spl_decode;
    
    MNN_PRINT("\n============== Speculative Decoding Statistics Start ===============\n");
    MNN_PRINT("Adopt speculative decode rate: %.2f%%\n", spl_rate);
    MNN_PRINT("Average speculative decode accept rate: %.2f%%\n", spl_accept_rate);

    // speculative decoding vs autoregressive decoding cost time rate
    // assume add 10% time with every additional token decoding
    float spl_cost_rate = 1.0 + (mDraftLength * 0.1);
    // original autoregressive decoding cost times
    float arg_time = spl_accept + arg_count;
    // speculative decoding cost times
    float spl_time = spl_count * spl_cost_rate + arg_count;
    float speed_up = 1.0 * arg_time / spl_time;
    MNN_PRINT("Draft length is: %d\n", mDraftLength);
    MNN_PRINT("Assume decode %d token is %.2f times consumption than single token decoding\n", mDraftLength, spl_cost_rate);
    MNN_PRINT("Total speed up is around: %.2fx\n", speed_up);
    MNN_PRINT("============== Speculative Decoding Statistics End =================\n");

#endif
    return;
}
    
} // namespace Transformer
} // namespace MNN
