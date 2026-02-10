#include "llm_benchmark_common.hpp"
#include <cstdio>

namespace MNN {
namespace Transformer {

void RunBenchmarkTest(Llm* llm, const LLMBenchMarkInstance& t) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    
    // Use fixed token ID 16, same as llm_bench.cpp
    int tok = 16;
    std::vector<int> tokens(t.n_prompt, tok);
    
    llm->generate(tokens, t.n_gen);
    auto context = llm->getContext();
    prompt_len += context->prompt_len;
    decode_len += context->gen_seq_len;
    prefill_time += context->prefill_us;
    decode_time += context->decode_us;
    
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("prompt tokens num = %d\n", prompt_len);
    printf("decode tokens num = %d\n", decode_len);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf("decode speed = %.2f tok/s\n", decode_len / decode_s);
}

} // namespace Transformer
} // namespace MNN
