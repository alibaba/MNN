//
// Created by ruoyi.sjd on 2024/12/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "llm_benchmark.hpp"
#include <cstdio>
#include "llm/llm.hpp"
#include <chrono>
#include <random>

using MNN::Transformer::Llm;

namespace mls {

static uint64_t GetTimeNs() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static void TestGen(Llm * llm, LLMBenchMarkInstance& t) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t vision_time = 0;
    int64_t audio_time = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    std::vector<int> ids{};
    ids.reserve(t.n_prompt);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10000);
    int random_number = distrib(gen);
    for (int i  = 0; i < t.n_prompt; i++) {
        ids.push_back(random_number);
    }
    llm->generate(ids, t.n_gen);
    // /*prompt_len += llm->prompt_len_;
    // decode_len += llm->gen_seq_len_;
    // vision_time += llm->vision_us_;
    // audio_time += llm->audio_us_;
    // prefill_time += llm->prefill_us_;
    // decode_time += llm->decode_us_;
    // float vision_s = vision_time / 1e6;
    // float audio_s = audio_time / 1e6;
    // float prefill_s = prefill_time / 1e6;
    // float decode_s = decode_time / 1e6;
    // printf("\n#################################\n");
    // printf("prompt tokens num = %d\n", prompt_len);
    // printf("decode tokens num = %d\n", decode_len);
    // printf(" vision time = %.2f s\n", vision_s);
    // printf("  audio time = %.2f s\n", audio_s);
    // printf("prefill time = %.2f s\n", prefill_s);
    // printf(" decode time = %.2f s\n", decode_s);
    // printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    // printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    // printf("##################################\n");*/
}

void LLMBenchmark::Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& params) {
    LLMBenchMarkInstance t {"smo", 128, 512};
    for (int i = 0; i < 5; i++) {
        TestGen(llm, t);
    }
}

}