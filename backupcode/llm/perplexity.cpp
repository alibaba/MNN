#include <algorithm>
#include <vector>
#include <cmath>
#include <evaluation/perplexity.hpp>
#include <llm/llm.hpp>
#include <iostream>
#include <iomanip>
#include <sampler/sampler.hpp>

namespace MNN{
namespace Transformer{

void PPLMeasurer::init(Llm* llm, std::vector<std::vector<int>> prompts, int max_len, int stride) {
    if (stride == 0) {
        // default stride for sliding window.
        stride = max_len / 2;
    }
    mLlm = llm;
    mMaxLen = max_len;
    mStride = stride;
    mPrompts = prompts;
}

PPLMeasurer::PPLMeasurer(Llm* llm, std::vector<std::vector<int>> prompts, int max_len, int stride) {
    init(llm, manager, prompts, max_len, stride);
}

PPLMeasurer::PPLMeasurer(Llm* llm, std::vector<std::string> prompts, int max_len, int stride) {
    std::vector<std::vector<int>> tokens(prompts.size());
    for (int p = 0; p < prompts.size(); ++p) tokens[p] = llm->encode(prompts[p]);
    init(llm, manager, tokens, max_len, stride);
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

float PPLMeasurer::perplexity_one(const std::vector<int>& prompt) {
    int seq_len = prompt.size();
    std::vector<float> nlls;
    float ppl = 0.f;
    
    // start calculation 
    int prev_end_loc = 1; // the first token start from id=1, do not count the first one.
    for (int begin_loc = 0; begin_loc < seq_len; begin_loc += mStride) {
        mStateCacheManager->setCurrentReference(mCandidate);
        int end_loc = std::min(begin_loc + mMaxLen, seq_len);
        // first token
        std::vector<int> tokens(prev_end_loc - begin_loc);
        for (int it = begin_loc; it < prev_end_loc; ++it) tokens[it - begin_loc] = prompt[it];
        auto logits = mLlm->forward(tokens, true);
        logits = MNN::Express::_Softmax(logits);
        nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[prev_end_loc]]));
        // std::cout << mLlm->decode(argmax(logits)) << "  " << mLlm->decode(prompt[prev_end_loc]) << "  " << -std::log(((float*)(logits->readMap<float>()))[prompt[prev_end_loc]]) << std::endl;
        // decode following tokens
        for (int it = prev_end_loc+1; it < end_loc; ++it) {
            auto logits = mLlm->forward({prompt[it-1]}, false);
            logits = MNN::Express::_Softmax(logits);
            nlls.push_back(-std::log(((float*)(logits->readMap<float>()))[prompt[it]]));
            // std::cout << mLlm->decode(argmax(logits)) << "  " << mLlm->decode(prompt[it]) << "  " << -std::log(((float*)(logits->readMap<float>()))[prompt[it]]) << std::endl;
        }
        // clean up once
        reset();
        mLlm->reset();
        prev_end_loc = end_loc;
        if (end_loc == seq_len) break;
    } 

    // calculate ppl
    for (int j = 0; j < nlls.size(); ++j) ppl += nlls[j];
    ppl /= nlls.size();
    ppl = std::exp(ppl);
    
    // print 
    std::cout << "PPL: " << std::setprecision(9) << ppl << std::endl;
    return ppl;
}

std::vector<float> PPLMeasurer::perplexity() {
    std::vector<float> ppls;
    for (auto prompt : mPrompts) {
        ppls.push_back(perplexity_one(prompt));
        reset();
        mLlm->reset();
    }
    return ppls;
}

void PPLMeasurer::reset() {
    // in the future, only reset its own.
}

void PPLMeasurer::reset(int max_len, int stride) {
    mMaxLen = max_len;
    mStride = stride;
    reset();
}

} // Transformer
} // MNN