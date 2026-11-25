//
//  bench.cpp
//  MNN
//
//  Created by MNN on 2026/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <llm/llm.hpp>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <streambuf>
#include <mach/mach.h>
using Clock = std::chrono::steady_clock;
using namespace MNN::Transformer;
class Capture : public std::streambuf {
public:
    Clock::time_point start, first;
    bool seen = false;
    std::string text;
    void record(const char* data, std::streamsize n) {
        if (n > 0) {
            if (!seen) {
                first = Clock::now();
                seen = true;
            }
            text.append(data, n);
        }
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        record(s, n);
        return n;
    }
    int overflow(int c) override {
        if (c != traits_type::eof()) {
            char ch = c;
            record(&ch, 1);
        }
        return traits_type::not_eof(c);
    }
};
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static void footprint() {
    task_vm_info_data_t vm = {};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &n) == KERN_SUCCESS) {
        printf("PEAK_FOOTPRINT %llu\n", static_cast<unsigned long long>(vm.ledger_phys_footprint_peak));
    }
}
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 3 && argc != 5) {
        fprintf(stderr, "usage: %s config.json export | config.json @128 rounds logits-prefix\n", argv[0]);
        return 2;
    }
    auto start = Clock::now();
    std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));
    if (!llm || !llm->load()) {
        return 3;
    }
    printf("LOAD_MS %.3f\n", ms(start, Clock::now()));
    if (argc == 3) {
        return std::string(argv[2]) == "export" ? 0 : 2;
    }
    const int rounds = atoi(argv[3]);
    if (rounds < 2) {
        return 2;
    }
    if (std::string(argv[2]) != "@128") {
        return 2;
    }
    std::string startText = "请阅读下面的英文单词：";
    const std::string endText = "。请详细介绍杭州的历史文化和著名景点，至少写三段。";
    for (int i = 0; i < 128 && llm->tokenizer_encode(llm->apply_chat_template(startText + endText)).size() < 128; ++i) {
        startText += " cat";
    }
    const std::string prompt = startText + endText;
    if (llm->tokenizer_encode(llm->apply_chat_template(prompt)).size() != 128) {
        return 7;
    }
    auto ctx = llm->getContext();
    for (int i = 0; i < rounds; ++i) {
        llm->reset();
        Capture capture;
        std::ostream stream(&capture);
        capture.start = Clock::now();
        llm->response(prompt, &stream, nullptr, 0);
        while (!llm->stoped() && ctx->gen_seq_len < 16) {
            llm->generate(1);
            if (ctx->status == LlmStatus::INTERNAL_ERROR) {
                return 5;
            }
        }
        auto end = Clock::now();
        if (!capture.seen || ctx->gen_seq_len != 16) {
            fprintf(stderr, "EARLY_EOS tokens=%d text=%s\n", ctx->gen_seq_len, capture.text.c_str());
            return 6;
        }

        printf(
            "BENCH "
            "{\"round\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,"
            "\"sample_ms\":%.3f,\"ttft_ms\":%.3f,\"e2e_ms\":%.3f,\"tokens\":[",
            i, ctx->prompt_len, ctx->gen_seq_len, ctx->prefill_us / 1000., ctx->decode_us / 1000.,
            ctx->sample_us / 1000., ms(capture.start, capture.first), ms(capture.start, end));
        for (size_t j = 0; j < ctx->output_tokens.size(); ++j) {
            printf("%s%d", j ? "," : "", ctx->output_tokens[j]);
        }
        printf("]}\n");
        fflush(stdout);
    }
    footprint();
    // Run a separate prefill so reading the full vocabulary never affects the timed requests.
    llm->reset();
    Capture capture;
    std::ostream stream(&capture);
    llm->response(prompt, &stream, nullptr, 0);
    auto outputs = llm->getOutputs();
    if (outputs.empty() || !outputs[0].get() || ctx->status == LlmStatus::INTERNAL_ERROR) {
        return 8;
    }
    auto value = outputs[0];
    const float* logits = value->readMap<float>();
    if (!logits || !value->getInfo()) {
        return 9;
    }
    std::ofstream matrix(std::string(argv[4]) + ".f32", std::ios::binary);
    matrix.write(reinterpret_cast<const char*>(logits), value->getInfo()->size * sizeof(float));
    matrix.close();
    return matrix ? 0 : 10;
}
