//
// Created by ruoyi.sjd on 2024/12/20.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
#pragma once
#include <string>
#include <vector>
#include "llm/llm.hpp"

namespace mls {

struct LLMBenchMarkInstance {
    std::string        model;//model name
    int                n_prompt;//prompt length
    int                n_gen;//gen length
    std::vector<uint64_t> samples_ns;//collected time ns
};

struct LLMBenchMarkOptions {
    bool progress;
    int reps;
};

//a benchmark referenced llama.cpp
class LLMBenchmark {
  public:
    void Start(MNN::Transformer::Llm* llm, const LLMBenchMarkOptions& benchmark_options);
};

}//mls