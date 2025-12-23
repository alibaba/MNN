//
//  mtp.cpp
//
//  Created by MNN on 2025/05/09.
//
//

#include "generate.hpp"

using namespace MNN::Express;
namespace MNN {
namespace Transformer {

MtpGeneration::MtpGeneration(Llm* llm, std::shared_ptr<LlmContext> context, std::shared_ptr<LlmConfig> config) : Generation(llm, context) {
    // do nothing
}
    
void MtpGeneration::load(Module::Config module_config) {
    mMtpMeta.reset(new KVMeta);
    mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMtpMeta.get());
    
    mMtpModules.resize(1);
    auto mtp_path = mLlm->mConfig->mtp_model();
    
    std::vector<std::string> inputNames{"input_embed", "hidden_states", "attention_mask", \
        "position_ids", "logits_index"};
    std::vector<std::string> outputNames {"logits"};
    mMtpModules[0].reset(Module::load(inputNames, outputNames, mtp_path.c_str(), mLlm->mRuntimeManager, &module_config));
    
    int verify_length = mLlm->mDraftLength + 1;
    // speculative decode module
    for(int i = 1; i <= verify_length; i++) {
        mMtpModulePool[std::make_pair(i, true)].reset(Module::clone(mMtpModules[0].get()));
    }
    // prefill module
    mMtpModulePool[std::make_pair(mLlm->mPrefillKey, false)] = mMtpModules[0];
    mHiddenStateIndex = mLlm->getOutputIndex("hidden_states");
}

std::vector<VARP> MtpGeneration::mtpForward(const std::vector<int>& input_ids, VARP hidden_states) {
    auto input_embeds = mLlm->embedding(input_ids);
    auto outputs = mtpForward(input_embeds, hidden_states);
    return outputs;
}
    
std::vector<VARP> MtpGeneration::mtpForward(Express::VARP input_embeds, VARP hidden_states) {
    int seq_len         = input_embeds->getInfo()->dim[0];
    mMtpMeta->add          = seq_len;
    auto attention_mask = mLlm->gen_attention_mask(seq_len);
    auto position_ids = mLlm->gen_position_ids(seq_len);

    VARP logitsIndex;
    bool inDecode = mContext->gen_seq_len > 0;
    bool isAllLogists = inDecode ? true : false;
    int seqLenKey = inDecode ? seq_len : mLlm->mPrefillKey;
    auto moduleKey = std::make_pair(seqLenKey, isAllLogists);
    
    if(mMtpModulePool.find(moduleKey) == mMtpModulePool.end()) {
        MNN_PRINT("Warning: mtp module need new clone, cloning now.\n");
        mLlm->mRuntimeManager->setHintPtr(Interpreter::KVCACHE_INFO, mMtpMeta.get());
        mMtpModulePool[moduleKey].reset(Module::clone(mMtpModules[0].get()));
    }
    
    // mtp decode get all index, prefill get l;ast index
    if (isAllLogists) {
        logitsIndex = mLlm->logitsAllIdx;
    } else {
        logitsIndex = mLlm->logitsLastIdx;
    }
    std::vector<Express::VARP> outputs;
    std::vector<Express::VARP> inputs = {input_embeds, hidden_states, attention_mask, \
        position_ids, logitsIndex};
    outputs = mMtpModulePool[moduleKey]->onForward(inputs);
    
    mMtpMeta->sync();
    
#if DEBUG_MODE == 3
    {
        std::ofstream outFile("input_embed.txt");
        auto temp = input_embeds->readMap<float>();
        for (size_t i = 0; i < input_embeds->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("hidden_states.txt");
        auto temp = hidden_states->readMap<float>();
        for (size_t i = 0; i < hidden_states->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("attention_mask.txt");
        auto temp = attention_mask->readMap<float>();
        for (size_t i = 0; i < attention_mask->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
    {
        std::ofstream outFile("position_ids.txt");
        auto temp = position_ids->readMap<int>();
        for (size_t i = 0; i < position_ids->getInfo()->size; ++i) {
            outFile << temp[i] << " "; // 每个数字后加空格
        }
        outFile.close();
    }
#endif
   
    return outputs;
}
    
    
void MtpGeneration::generate(GenerationParams& param) {
    int max_token = param.max_new_tokens;
    VARP input_embeds = param.input_embeds;
    if(input_embeds == nullptr && !param.input_ids.empty()) {
        auto input_ids = param.input_ids;
        std::vector<int> current_ids(input_ids.begin()+1, input_ids.end());
        current_ids.push_back(mContext->current_token);
        input_embeds = mLlm->embedding(current_ids);
    }
    VARP prev_hidden_states = param.outputs[mHiddenStateIndex];
    
    // generate first draft
    std::vector<int> mtp_draft(mLlm->mDraftLength);
    {
        mMtpMeta->remove = 0;
        auto cur_embed = mLlm->embedding({mContext->current_token});
        auto pre_embeds = _Split(input_embeds, {1, input_embeds->getInfo()->dim[0]-1}, 0);
        auto prefill_embeds = _Concat({pre_embeds[1], cur_embed}, 0);

        auto mtpDraft = mtpForward(prefill_embeds, prev_hidden_states);
        
        auto sample_size = mtpDraft[0]->getInfo()->dim[mtpDraft[0]->getInfo()->dim.size() - 1];
        for(int i = 0; i < mLlm->mDraftLength; i++) {
            auto sample_offset = i * sample_size;
            mtp_draft[i] = mLlm->sample(mtpDraft[0], sample_offset, sample_size);
        }
    }

    bool stop = false;
    int len = 0;
    // speculative total decode token numbers
    int spl_decode = 0;
    // speculative accept token numbers
    int spl_accept = 0;
    // speculative number of times
    int spl_count = 0;

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
            drafts.insert(drafts.end(), mtp_draft.begin(), mtp_draft.end());
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
            if (outputs.size() < 2) {
                break;
            }
            auto logits = outputs[0];
            if (logits->getInfo()->size == 0) {
                break;
            }

            // verify draft token whether be accepted
            int i_dft = draftVerify(logits, drafts, stop);
            
            // MTP Draft Generate
            {
                AUTOTIME;
                std::vector<int> currentIds;
                for(int i = 1; i < i_dft; i++) {
                    currentIds.push_back(drafts[i]);
                }
                currentIds.push_back(mContext->current_token);
                auto prev_hidden_states = outputs[1];
                auto mtpDraft = mtpForward(currentIds, prev_hidden_states);
                mMtpMeta->remove         = drafts.size() - i_dft;

                auto sample_size = mtpDraft[0]->getInfo()->dim[mtpDraft[0]->getInfo()->dim.size() - 1];
                
                for(int i = 0; i < mLlm->mDraftLength; i++) {
                    auto sample_offset = (i * mtpDraft[0]->getInfo()->dim[1] + i_dft - 1) * sample_size;
                    mtp_draft[i] = mLlm->sample(mtpDraft[0], sample_offset, sample_size);
                }
                
            }
            // clear dirty kv-cache
            mLlm->mMeta->remove = drafts.size() - i_dft;
            len += i_dft;

            // update context state
            mLlm->updateContext(i_dft, i_dft-1);
            
            // count time cost
            mContext->decode_us += _t.durationInUs();

            // add all accept tokens to string
            mContext->history_tokens.insert(mContext->history_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            mContext->output_tokens.insert(mContext->output_tokens.end(), drafts.begin(), drafts.begin() + i_dft);
            
        #ifdef DUMP_PROFILE_INFO
            spl_decode += drafts.size();
            spl_accept += i_dft;
            spl_count++;
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
    // draft accept rate if adopt speculative decoding
    float spl_accept_rate = 100.0 * spl_accept / spl_decode;
    int verify_length = mLlm->mDraftLength + 1;

    MNN_PRINT("\n============== MTP Decoding Statistics Start ===============\n");
    MNN_PRINT("Average mtp decode accept rate: %.2f%%\n", spl_accept_rate);
    MNN_PRINT("Verify length is: %d\n", verify_length);

    // speculative decoding vs autoregressive decoding cost time rate
    // assume add 10% time with every additional token decoding
    float overhead = 0.2;
    for(int i = 0; i < 3; i++) {
        float spl_cost_rate = 1.0 + (verify_length * overhead);
        // original autoregressive decoding cost times
        float arg_time = spl_accept;
        // speculative decoding cost times
        float spl_time = spl_count * spl_cost_rate;
        // add mtp head cost time, assume mtp need 10% main cost each mtp layer
        spl_time *= spl_cost_rate;
        float speed_up = 1.0 * arg_time / spl_time;
        MNN_PRINT("\nIf assume0: decode each mtp head need %.2f%% of main model cost\n", overhead*100);
        MNN_PRINT("If assume1: decode %d token is %.2f times consumption than single token decoding\n", verify_length, spl_cost_rate);
        MNN_PRINT("Total speed up is around: %.2fx\n", speed_up);
        
        overhead -= 0.05;
    }
    MNN_PRINT("============== MTP Decoding Statistics End =================\n");

#endif
    return;
}

    
} // namespace Transformer
} // namespace MNN
