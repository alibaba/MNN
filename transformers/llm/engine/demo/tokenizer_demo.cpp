//
//  tokenizer_demo.cpp
//
//  Created by MNN on 2025/09/01.
//  ZhaodeWang
//

#include "../src/tokenizer/tokenizer.hpp"
#include <chrono>

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt [bench|test]" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::string mode = (argc >= 3) ? argv[2] : "";

    if (mode == "bench") {
        // Benchmark mode: measure load time and encode/decode speed
        int rounds = 5;
        // 1. Load benchmark
        double load_total = 0;
        Tokenizer* tok = nullptr;
        for (int i = 0; i < rounds; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            tok = Tokenizer::createTokenizer(tokenizer_path);
            auto t1 = std::chrono::high_resolution_clock::now();
            load_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (i < rounds - 1) delete tok;
        }
        printf("Load:   avg %.2f ms (%d rounds)\n", load_total / rounds, rounds);
        std::unique_ptr<Tokenizer> tokenizer(tok);

        // 2. Encode benchmark (short strings)
        std::vector<std::string> bench_strs = {
            "介绍一下北京的首都",
            "Hello World, this is a test of tokenizer performance.",
            "The quick brown fox jumps over the lazy dog. 1234567890!@#$%",
            "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质。",
        };
        int encode_rounds = 100;
        double encode_total = 0;
        int total_tokens = 0;
        for (int r = 0; r < encode_rounds; r++) {
            for (auto& s : bench_strs) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto ids = tokenizer->encode(s);
                auto t1 = std::chrono::high_resolution_clock::now();
                encode_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (r == 0) total_tokens += ids.size();
            }
        }
        printf("Encode(short): avg %.3f ms / call (%d strings x %d rounds, %d tokens/round)\n",
               encode_total / (encode_rounds * bench_strs.size()), (int)bench_strs.size(), encode_rounds, total_tokens);

        // 2b. Encode benchmark (long ~5K string)
        std::string long_text;
        {
            std::string block = "在人工智能领域中，大型语言模型（Large Language Model, LLM）是一种基于深度学习的自然语言处理技术。"
                "These models are trained on massive datasets containing billions of tokens from diverse sources including books, websites, and academic papers. "
                "模型通过自注意力机制（Self-Attention）来捕捉文本中长距离的依赖关系，从而实现对语言的深层理解。"
                "The transformer architecture, introduced in the seminal paper 'Attention Is All You Need', revolutionized the field of NLP. "
                "在实际应用中，LLM被广泛用于对话系统、代码生成、文本摘要、翻译等多种任务。1234567890!@#$%^&*() ";
            while (long_text.size() < 5000) long_text += block;
        }
        int long_rounds = 20;
        double long_encode_total = 0;
        int long_tokens = 0;
        for (int r = 0; r < long_rounds; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto ids = tokenizer->encode(long_text);
            auto t1 = std::chrono::high_resolution_clock::now();
            long_encode_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (r == 0) long_tokens = (int)ids.size();
        }
        printf("Encode(long):  avg %.3f ms / call (%d chars, %d tokens, %d rounds)\n",
               long_encode_total / long_rounds, (int)long_text.size(), long_tokens, long_rounds);

        // 3. Decode benchmark
        auto sample_ids = tokenizer->encode(bench_strs[3]);
        int decode_rounds = 1000;
        double decode_total = 0;
        for (int r = 0; r < decode_rounds; r++) {
            for (auto id : sample_ids) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto s = tokenizer->decode(id);
                auto t1 = std::chrono::high_resolution_clock::now();
                decode_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }
        printf("Decode: avg %.4f ms / token (%d tokens x %d rounds)\n",
               decode_total / (decode_rounds * sample_ids.size()), (int)sample_ids.size(), decode_rounds);
        return 0;
    }

    // Default mode: encode + decode correctness test
    auto t0 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<Tokenizer> tokenizer(Tokenizer::createTokenizer(tokenizer_path));
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Load time: %.2f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    std::vector<std::string> test_strs = {"介绍一下北京的首都", "Hello World", "The quick brown fox"};
    for (auto& s : test_strs) {
        auto ids = tokenizer->encode(s);
        std::string decoded;
        for (auto id : ids) decoded += tokenizer->decode(id);
        bool ok = (decoded == s);
        printf("[%s] \"%s\" -> encode -> decode -> \"%s\"\n", ok ? "PASS" : "FAIL", s.c_str(), decoded.c_str());
    }
    return 0;
}
