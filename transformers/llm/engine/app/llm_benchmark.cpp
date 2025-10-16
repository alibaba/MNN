//
// Created by ruoyi.sjd on 2024/12/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//
#include "llm_benchmark.hpp"
#include "llm_benchmark_common.hpp"

using MNN::Transformer::Llm;

namespace mls {

void LLMBenchmark::Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& params) {
    MNN::Transformer::LLMBenchMarkInstance t {"", 128, 512};
    for (int i = 0; i < 1; i++) {
        MNN::Transformer::RunBenchmarkTest(llm, t);
    }
}

}