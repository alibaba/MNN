#include <algorithm>
#include <vector>
#include <cmath>
#include <llm/llm.hpp>
#include <iostream>
#include <iomanip>

#include "sampler.hpp"
#include "perplexity.hpp"
#include "llmconfig.hpp"
#include "prompt.hpp"

namespace MNN{
namespace Transformer{


/* -----------TextPPLMeasurer---------- */
TextPPLMeasurer::TextPPLMeasurer(Llm* llm, std::shared_ptr<LlmConfig> llmConfig) {
    mLlm = llm;
    mConfig.max_all_tokens = llmConfig->max_all_tokens();
    mConfig.max_new_tokens = llmConfig->max_new_tokens();
    mDatasetType = llmConfig->dataset();
    mStride = llmConfig->ppl_stride();
    if (mStride == 0) {
        // default stride for sliding window.
        mStride = mConfig.max_all_tokens / 2;
    } 
}

/* Implemented based on https://huggingface.co/docs/transformers/perplexity

 ******************** HuggingFace Python Version ************************

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())

 ******************** HuggingFace Python Version ************************ 
*/

float TextPPLMeasurer::perplexity_one(const std::vector<int>& prompt) {
    int seq_len = prompt.size();
    std::vector<float> nlls;
    float ppl = 0.f;
    
    // start calculation 
    int prev_end_loc = 1; // the first token start from id=1, do not count the first one.
    for (int begin_loc = 0; begin_loc < seq_len; begin_loc += mStride) {
        int end_loc = std::min(begin_loc + mConfig.max_all_tokens, seq_len);
        // first token
        std::vector<int> tokens(prev_end_loc - begin_loc);
        for (int it = begin_loc; it < prev_end_loc; ++it) tokens[it - begin_loc] = prompt[it];
        mLlm->mLlmSessionInfos[0].all_seq_len_ = tokens.size();
        mLlm->mLlmSessionInfos[0].gen_seq_len_ = mLlm->mLlmSessionInfos[0].all_seq_len_;
        auto logits = mLlm->forward(tokens, mLlm->mLlmSessionInfos[0].all_seq_len_, mLlm->mLlmSessionInfos[0].gen_seq_len_, true);
        logits = MNN::Express::_Softmax(logits);
        nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[prev_end_loc]]));
        // std::cout << mLlm->decode(argmax(logits)) << "  " << mLlm->decode(prompt[prev_end_loc]) << "  " << -std::log(((float*)(logits->readMap<float>()))[prompt[prev_end_loc]]) << std::endl;
        std::cout << -std::log(((float*)(logits->readMap<float>()))[prompt[prev_end_loc]]) << std::endl;
        // decode following tokens
        for (int it = prev_end_loc+1; it < end_loc; ++it) {
            mLlm->mLlmSessionInfos[0].all_seq_len_ += 1;
            mLlm->mLlmSessionInfos[0].gen_seq_len_ = mLlm->mLlmSessionInfos[0].all_seq_len_;
            auto logits = mLlm->forward({prompt[it-1]},mLlm->mLlmSessionInfos[0].all_seq_len_, mLlm->mLlmSessionInfos[0].gen_seq_len_, false);
            logits = MNN::Express::_Softmax(logits);
            nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[it]]));
            // std::cout << mLlm->decode(argmax(logits)) << "  " << mLlm->decode(prompt[it]) << "  " << -std::log(((float*)(logits->readMap<float>()))[prompt[it]]) << std::endl;
            std::cout << -std::log(((float*)(logits->readMap<float>()))[prompt[it]]) << std::endl;
        }
        // clean up once
        mLlm->reset();
        prev_end_loc = end_loc;
        if (end_loc == seq_len) break;
    } 

    // calculate ppl
    for (int j = 0; j < nlls.size(); ++j) ppl += nlls[j];
    ppl /= nlls.size();
    ppl = std::exp(ppl);
    
    // print 
    std::cout << "PPL: " << std::setprecision(8) << ppl << std::endl;
    return ppl;
}

std::vector<float> TextPPLMeasurer::perplexity(std::vector<std::vector<int>> prompts) {
    std::vector<float> ppls;
    for (auto prompt : prompts) {
        ppls.push_back(perplexity_one(prompt));
        mLlm->reset();
    }
    return ppls;
}

std::vector<float> TextPPLMeasurer::perplexity(std::vector<std::string> prompts) {
    std::vector<std::vector<int>> tokens(prompts.size());
    for (int p = 0; p < prompts.size(); ++p) tokens[p] = mLlm->tokenizer(prompts[p]);
    return perplexity(tokens);
}

std::vector<float> TextPPLMeasurer::perplexity(std::string prompt_file, std::ostream* perfOS) {
    // No performance will be printed!
    std::vector<std::string> prompts;
    if (mDatasetType == "wikitext") {
        prompts = wikitext(prompt_file);
    }
    else if (mDatasetType == "plaintext") {
        prompts = plaintext(prompt_file);
    }
    else if (mDatasetType == "rowsplit") {
        prompts = rowsplit(prompt_file);
    }
    else {
        MNN_ERROR("Dataset not suppoted");
        exit(1);
    }
    std::cout << "prompt file loaded!" << std::endl;
    return perplexity(prompts);
}

/* -----------ChatPPLMeasurer---------- */
ChatPPLMeasurer::ChatPPLMeasurer(Llm* llm, std::shared_ptr<LlmConfig> llmConfig) {
    mLlm = llm;
    mConfig.max_all_tokens = llmConfig->max_all_tokens();
    mConfig.max_new_tokens = llmConfig->max_new_tokens();
    mDatasetType = llmConfig->dataset();
    mDatasetSampleSize = llmConfig->dataset_sample_size();
}

void ChatPPLMeasurer::handleToken(int token) {
    // CommonPrefix and Candidates managements
    mLlm->mLlmSessionInfos[0].tokens.push_back(token);
    mLlm->mLlmSessionInfos[0].all_seq_len_++;
    mLlm->mLlmSessionInfos[0].gen_seq_len_++;
}

std::vector<float> ChatPPLMeasurer::sample(const std::vector<int>& input_ids, const std::vector<int>& prompt, struct TimePerformance* time_perf) {
    std::vector<float> nlls;
    // initialization for time performance
    PrefillTimePerformance prefill_time;
    prefill_time.prefill_prev_token_ = mLlm->mLlmSessionInfos[0].tokens.size();
    prefill_time.prefill_token_ = input_ids.size();
    appendNewPromptRecord(time_perf, input_ids.size(), mLlm->reuse_kv());
    // initialization
    mLlm->mLlmSessionInfos[0].tokens.insert(mLlm->mLlmSessionInfos[0].tokens.end(), input_ids.begin(), input_ids.end());
    // all_seq_len_ in sampler functions as kv_seq_len_, prev_seq_len_ = all_seq_len_ - seq_len
    mLlm->mLlmSessionInfos[0].all_seq_len_ = mLlm->mLlmSessionInfos[0].tokens.size(); 
    mLlm->mLlmSessionInfos[0].gen_seq_len_ = 0;
    // prefill 
    auto st = std::chrono::system_clock::now();
    auto logits = mLlm->forward(input_ids, mLlm->mLlmSessionInfos[0].all_seq_len_, mLlm->mLlmSessionInfos[0].gen_seq_len_, true);
    logits = MNN::Express::_Softmax(logits);
    nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[mLlm->mLlmSessionInfos[0].gen_seq_len_]]));
    // record time
    auto et = std::chrono::system_clock::now();
    prefill_time.prefill_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
    time_perf->prefill_record_.push_back(prefill_time);
    // handle the new token
    handleToken(prompt[mLlm->mLlmSessionInfos[0].gen_seq_len_]);
    // decode
    while (mLlm->mLlmSessionInfos[0].gen_seq_len_ < prompt.size()) {
        DecodeTimePerformance decode_time;
        decode_time.decode_prev_token_ = mLlm->mLlmSessionInfos[0].tokens.size();
        st = std::chrono::system_clock::now();
        // next token
        logits = mLlm->forward({mLlm->mLlmSessionInfos[0].tokens.back()}, mLlm->mLlmSessionInfos[0].all_seq_len_, mLlm->mLlmSessionInfos[0].gen_seq_len_, false);
        logits = MNN::Express::_Softmax(logits);
        nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[mLlm->mLlmSessionInfos[0].gen_seq_len_]]));
        et = std::chrono::system_clock::now();
        decode_time.decode_us_ = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
        time_perf->decode_record_.push_back(decode_time);
        handleToken(prompt[mLlm->mLlmSessionInfos[0].gen_seq_len_]);
    }
    // return nlls
    return nlls;
}

float ChatPPLMeasurer::perplexity_one(const std::vector<std::vector<PromptItem>>& prompt, std::ostream* perfOS) {
    // (turns, roles)
    std::vector<float> nlls;
    float ppl = 0.f;

    // < simulated chat
    mLlm->reset();
    for (auto& turn : prompt) {
        mLlm->mPromptLib->appendUserPrompt(turn[0].second);
        std::vector<int> input_ids = mLlm->tokenizer(mLlm->mPromptLib->getLLMInput());
        mLlm->generate_init();
        auto turn_nlls = sample(input_ids, mLlm->tokenizer(turn[1].second), &(mLlm->mLlmSessionInfos[0].mTimePerformance));
        nlls.insert(nlls.end(), turn_nlls.begin(), turn_nlls.end());
        mLlm->mPromptLib->appendLLMOutput(turn[1].second);
    }

    // record time performance to file
    if (perfOS != nullptr) {
        mLlm->mLlmSessionInfos[0].print_speed(perfOS);
    }

    mLlm->reset();
    // simulated chat >

    // calculate ppl
    for (int j = 0; j < nlls.size(); ++j) ppl += nlls[j];
    ppl /= nlls.size();
    ppl = std::exp(ppl);
    
    // print 
    std::cout << "PPL: " << std::setprecision(8) << ppl << std::endl;
    return ppl;
}


std::vector<float> ChatPPLMeasurer::perplexity(const std::vector<std::vector<std::vector<PromptItem>>>& prompts, std::ostream* perfOS) {
    std::vector<float> ppls;
    for (auto& prompt : prompts) {
        ppls.push_back(perplexity_one(prompt, perfOS));
        mLlm->reset();
    }
    return ppls;
}

void ChatPPLMeasurer::getStats(const std::vector<std::vector<std::vector<PromptItem>>>& prompts) {
    std::ofstream total_stats("total_stats.csv");
    std::ofstream dialog_stats("dialog_stats.csv");
    float average_turns=0, average_prefill=0, average_decode=0, average_total_tokens=0;
    int max_turns=0;
    std::vector<std::vector<std::vector<int>>> stats; // (dialog, turn, (prefill, decode))
    std::cout << prompts.size() << std::endl;
    int counter = 0;
    for (auto& dialog : prompts) {
        std::vector<std::vector<int>> dialog_stats;
        if ((counter++) % std::max((int)prompts.size()/200, 1) == 0) std::cout << "*" << std::flush;
        float prefill_len_turn = 0;
        float decode_len_turn = 0;
        for (auto& turn : dialog) {
            // turn: prefill, decode
            int prefill_len = mLlm->tokenizer(turn[0].second).size();
            int decode_len = mLlm->tokenizer(turn[1].second).size();
            prefill_len_turn += prefill_len;
            decode_len_turn += decode_len;
            average_total_tokens += prefill_len + decode_len;
            dialog_stats.push_back({prefill_len, decode_len});
        }
        stats.push_back(dialog_stats);
        average_prefill += prefill_len_turn / dialog.size(); // average over turns
        average_decode += decode_len_turn / dialog.size(); // average over turns
        average_turns += dialog.size();
        max_turns = std::max(max_turns, (int)dialog.size());
    }
    average_turns /= prompts.size();
    average_prefill /= prompts.size();
    average_decode /= prompts.size();
    average_total_tokens /= prompts.size();
    total_stats << "total_dialogs," << "max_turns," << "avg_turns," \
                 << "avg_prefill_tokens/turn," << "avg_decode_tokens/turn," \
                  << "avg_total_tokens/dialog" << std::endl;
    total_stats << prompts.size() << ","  << max_turns << "," << average_turns << "," \
                 << average_prefill << "," << average_decode << "," \
                  << average_total_tokens << std::endl;
    for (int i=0; i<max_turns; ++i) dialog_stats <<  "prefill" << i << "," << "decode" << i << ","; // this creates an extra blank column at the end.
    dialog_stats <<  std::endl;
    for (auto& dialog : stats) {
        for (auto& turn : dialog){
            dialog_stats << turn[0] << "," << turn[1] << ",";
        }
        for (int i=dialog.size(); i<max_turns; ++i) {
            dialog_stats <<  ",,";
        }  
        dialog_stats <<  std::endl;
    }
}   


std::vector<float> ChatPPLMeasurer::perplexity(std::string prompt_file, std::ostream* perfOS) {
    // No performance will be printed!
    std::vector<std::vector<std::vector<PromptItem>>> prompts;
    if (mDatasetType == "shareGPT") {
        prompts = shareGPT(prompt_file, mDatasetSampleSize);
    }
    else {
        MNN_ERROR("Dataset not suppoted");
        exit(1);
    }
    std::cout << "prompt file loaded!" << std::endl;
    getStats(prompts);
    std::cout << "\nshareGPT statistics counted!" << std::endl;
    return perplexity(prompts, perfOS);
}


} // Transformer
} // MNN