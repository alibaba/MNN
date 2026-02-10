//
//  lookahead.cpp
//
//  Created by MNN on 2025/04/09.
//

#include "ngram.hpp"
#include "lookahead.hpp"
#include "generate.hpp"

//#define DUMP_PROFILE_INFO

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

LookaheadGeneration::LookaheadGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    mNgramKeyMaxLen = config->ngram_match_maxlen();
    if(mNgramKeyMaxLen > 8) {
        MNN_PRINT("Warning: ngram match key length maybe too large!\n");
    }
    auto strictness = config->draft_match_strictness();
    mStrictLevel = MatchStrictLevel::LOW_LEVEL;
    if(strictness == "high") {
        mStrictLevel = MatchStrictLevel::HIGH_LEVEL;
    } else if(strictness == "medium") {
        mStrictLevel = MatchStrictLevel::MEDIUM_LEVEL;
    } else if(strictness == "low"){
        mStrictLevel = MatchStrictLevel::LOW_LEVEL;
    } else {
        MNN_PRINT("Warning: draft_match_strictness value set error!, use default param instead\n");
    }
    
    auto selectRule = config->draft_selection_rule();
    mSelectRule = NgramSelectRule::FreqxLen_RULE;
    if(selectRule == "fcfs") {
        mSelectRule = NgramSelectRule::FCFS_RULE;
    } else if(selectRule == "freqxlen"){
        mSelectRule = NgramSelectRule::FreqxLen_RULE;
    } else {
        MNN_PRINT("Warning: draft_selection_rule value set error!, use default param instead\n");
    }
    mUpdateNgram = config->ngram_update();
}
    
void LookaheadGeneration::generate(GenerationParams& param) {
    if (-1 == mContext->current_token) {
        mContext->current_token = mLlm->sample(param.outputs[0]);
    }
    int max_token = param.max_new_tokens;
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
        auto prior_prompt_file = mLlm->mConfig->lookup_file();
        std::ifstream file(prior_prompt_file);

        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string user_set_prompt = buffer.str();
            file.close();
            std::vector<int> user_ids = mLlm->tokenizer_encode(user_set_prompt);
            
            if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            } else {
                ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, user_ids, user_ids.size());
            }
        }
    }
    bool stop = false;
    // speculative total decode token numbers
    int spl_decode = 0;
    // speculative accept token numbers
    int spl_accept = 0;
    // autoregression number of times
    int arg_count = 0;
    // speculative number of times
    int spl_count = 0;
    int verify_len = mLlm->mDraftLength + 1;
    
    while (len < max_token) {
        if(mContext->status == LlmStatus::USER_CANCEL) {
            break;
        }
        MNN::Timer _t;
        std::vector<int> drafts;
        drafts.push_back(mContext->current_token);
        
        auto decodeStr = mLlm->tokenizer_decode(mContext->current_token);
        mContext->generate_str += decodeStr;
        if (nullptr != mContext->os) {
            *mContext->os << decodeStr;
            *mContext->os << std::flush;
        }
        // mContext->current_token add to gen_seq_len
        mLlm->updateContext(0, 1);

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
                ngram_cache_search(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts, verify_len, level);
            } else {
                ngram_cache_search(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, drafts, verify_len, level);
            }
            
            mLlm->mMeta->add = drafts.size();

            AUTOTIME;
            // do draft token parallel verify
            auto outputs = mLlm->forwardVec(drafts);
            for (auto o : outputs) {
                if(nullptr == o->readMap<float>()) {
                    mContext->status = LlmStatus::INTERNAL_ERROR;
                    break;
                }
            }
            if(outputs.empty()) {
                break;
            }
            auto logits = outputs[0];
            if (nullptr == logits.get()) {
                break;
            }
            if (logits->getInfo()->size == 0) {
                break;
            }

            // verify draft token whether be accepted
            int i_dft = draftVerify(logits, drafts, stop);

            // update ngram for each decoding
            if(!stop && mUpdateNgram) {
                if(mSelectRule == NgramSelectRule::FreqxLen_RULE) {
                    ngram_cache_update(prompt_ngram_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                } else {
                    ngram_cache_update(prompt_ngram_ordered_cache, 1, mNgramKeyMaxLen, mContext->history_tokens, i_dft);
                }
            }

            // clear dirty kv-cache
            mLlm->mMeta->remove = drafts.size() - i_dft;
            len += i_dft;
            
            // update context state
            int seq_len = i_dft;
            int gen_len = i_dft - 1; // current_token has been added, add others
            mLlm->updateContext(seq_len, gen_len);
            
            // count time cost
            mContext->decode_us += _t.durationInUs();

            // add all accept tokens to string
            mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            mContext->output_tokens.insert(mContext->output_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            
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
            if(stop) {
                mContext->history_tokens.push_back(mContext->current_token);
                mContext->output_tokens.push_back(mContext->current_token);
                mLlm->updateContext(0, 1);
                break;
            }
            if (mLlm->is_stop(mContext->current_token)) {
                mContext->history_tokens.push_back(mContext->current_token);
                mContext->output_tokens.push_back(mContext->current_token);
                mLlm->updateContext(0, 1);
                if (nullptr != mContext->os) {
                    *mContext->os << mContext->end_with << std::flush;
                }
                break;
            }
        }
    }
    if(len >= max_token) {
        mContext->status = LlmStatus::MAX_TOKENS_FINISHED;
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
    float spl_cost_rate = 1.0 + (verify_len * 0.1);
    // original autoregressive decoding cost times
    float arg_time = spl_accept + arg_count;
    // speculative decoding cost times
    float spl_time = spl_count * spl_cost_rate + arg_count;
    float speed_up = 1.0 * arg_time / spl_time;
    MNN_PRINT("Verify length is: %d\n", verify_len);
    MNN_PRINT("Assume decode %d token is %.2f times consumption than single token decoding\n", verify_len, spl_cost_rate);
    MNN_PRINT("Total speed up is around: %.2fx\n", speed_up);
    MNN_PRINT("============== Speculative Decoding Statistics End =================\n");

#endif
    return;
}
    
} // namespace Transformer
} // namespace MNN
