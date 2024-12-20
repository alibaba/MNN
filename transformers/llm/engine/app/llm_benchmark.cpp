//
// Created by ruoyi.sjd on 2024/12/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "llm_benchmark.hpp"
#include <cstdio>
#include "llm/llm.hpp"
#include <chrono>

using MNN::Transformer::Llm;

namespace mls {

static uint64_t GetTimeNs() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static void TestGen(Llm * llm, int n_gen, int n_threads) {
    std::vector<int> ids = {1,3,5,49151};
    llm->generate(ids);
}

//modified from the llama.cpp benchmark
void LLMBenchmark::Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& params) {
    LLMBenchMarkInstance t {"smo", 100, 512};
    TestGen(llm, t.n_gen, 1);
//    if (t.n_prompt > 0) {
////        if (params.progress) {
////            fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup prompt run\n", params_idx, params_count);
////        }
//        //test_prompt(ctx, std::min(t.n_batch, std::min(t.n_prompt, 32)), 0, t.n_batch, t.n_threads);
//        test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
//    }
//    if (t.n_gen > 0) {
//        if (params.progress) {
//            fprintf(stderr, "llama-bench: benchmark %d/%zu: warmup generation run\n", params_idx, params_count);
//        }
//        test_gen(ctx, 1, t.n_threads);
//    }
//
//    for (int i = 0; i < params.reps; i++) {
//        llama_kv_cache_clear(ctx);
//
//        uint64_t t_start = GetTimeNs();
//
////        if (t.n_prompt > 0) {
////            if (params.progress) {
////                fprintf(stderr, "llama-bench: benchmark %d/%zu: prompt run %d/%d\n", params_idx, params_count,
////                        i + 1, params.reps);
////            }
////            test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads);
////        }
//        if (t.n_gen > 0) {
//            test_gen(ctx, t.n_gen, t.n_threads);
//        }
//
//        uint64_t t_ns = get_time_ns() - t_start;
//        t.samples_ns.push_back(t_ns);
//    }
}

}