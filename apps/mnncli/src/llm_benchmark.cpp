//
// Created by ruoyi.sjd on 2024/12/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "llm_benchmark.hpp"
#include "../../../transformers/llm/engine/app/llm_benchmark_common.hpp"

using MNN::Transformer::Llm;

namespace mnncli {

void LLMBenchmark::Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& params) {
    printf("\n=== LLM Benchmark ===\n");
    printf("Benchmark parameters:\n");
    printf("  - Prompt length: 128 tokens\n");
    printf("  - Generation length: 512 tokens\n");
    printf("  - Token ID: 16 (fixed)\n");
    printf("Starting benchmark...\n\n");
    
    MNN::Transformer::LLMBenchMarkInstance t {"", 128, 512};
    for (int i = 0; i < 1; i++) {
        MNN::Transformer::RunBenchmarkTest(llm, t);
    }
    
    printf("\n=== Benchmark Complete ===\n");
}

}