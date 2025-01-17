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
    prompt_len += llm->getState().prompt_len_;
    decode_len += llm->getState().gen_seq_len_;
    prefill_time += llm->getState().prefill_us_;
    decode_time += llm->getState().decode_us_;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("prompt tokens num = %d\n", prompt_len);
    printf("decode tokens num = %d\n", decode_len);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf("decode speed = %.2f tok/s\n", decode_len / decode_s);
}

void LLMBenchmark::Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& params) {
    LLMBenchMarkInstance t {"", 128, 512};
    for (int i = 0; i < 1; i++) {
        TestGen(llm, t);
    }
}

}