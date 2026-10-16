// End-to-end greedy LLM smoke test. A graph recording count alone does not imply NX hardware execution.
#include "component/VulkanArmNx.hpp"
#include <llm/llm.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s config.json prompt [max_tokens=24] [logits.bin] [--require-arm-nx] [--teacher-tokens file]\n",
               argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int maxTokens = argc > 3 ? atoi(argv[3]) : 24;
    std::unique_ptr<MNN::Transformer::Llm> llm(MNN::Transformer::Llm::createLLM(argv[1]));
    llm->set_config(
        R"({"async":false,"all_logits":false,"enable_debug":true,"jinja":{"context":{"enable_thinking":false}}})");
    if (!llm->load())
        return 1;
    uint64_t convs = 0, fused = 0;
    llm->setDebugCallback([](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo*) { return true; },
                          [&](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* op) {
                              if (op->type() == "Convolution")
                                  ++convs;
                              if (op->type() == "FusedProj")
                                  ++fused;
                              return true;
                          });
    auto ids = llm->tokenizer_encode(llm->apply_chat_template(argv[2]));
    printf("Prompt tokens=%zu; generating at most %d tokens\n", ids.size(), maxTokens);
    llm->generate_init();
    std::ofstream dump;
    bool requireArmNx = false;
    std::vector<int> teacherTokens;
    for (int i = 4; i < argc; ++i) {
        if (!strcmp(argv[i], "--require-arm-nx"))
            requireArmNx = true;
        else if (!strcmp(argv[i], "--teacher-tokens") && i + 1 < argc) {
            std::ifstream input(argv[++i]);
            int token;
            while (input >> token)
                teacherTokens.push_back(token);
            if (teacherTokens.size() < static_cast<size_t>(maxTokens))
                return 1;
        } else
            dump.open(argv[i], std::ios::binary);
    }
    for (int step = 0; step < maxTokens; ++step) {
        const auto before = MNN::VulkanArmNx::recordedDispatchCount();
        const auto convsBefore = convs;
        auto logits = llm->forward(ids, step == 0);
        if (logits == nullptr)
            return 2;
        const auto* values = logits->readMap<float>();
        if (!values)
            return 2;
        const int vocab = logits->getInfo()->dim.back();
        int token = 0;
        for (int i = 0; i < vocab; ++i) {
            if (!std::isfinite(values[i]))
                return 3;
            if (values[i] > values[token])
                token = i;
        }
        const auto graphs = MNN::VulkanArmNx::recordedDispatchCount() - before;
        if (requireArmNx && step < 2 && (graphs == 0 || graphs < convs - convsBefore)) {
            fprintf(stderr, "Required ARM NX Linear graph coverage was not reached at step %d\n", step);
            return 4;
        }
        if (dump.is_open())
            dump.write(reinterpret_cast<const char*>(values), vocab * sizeof(float));
        printf("STEP %d token=%d graphs_recorded=%llu convs=%llu fused=%llu text=%s\n", step, token,
               static_cast<unsigned long long>(MNN::VulkanArmNx::recordedDispatchCount() - before),
               static_cast<unsigned long long>(convs), static_cast<unsigned long long>(fused),
               llm->tokenizer_decode(token).c_str());
        const int next = teacherTokens.empty() ? token : teacherTokens[step];
        if (llm->is_stop(next))
            break;
        ids = {next};
    }
    printf("PASS: finite LLM logits; ARM NX graph recordings=%llu\n",
           static_cast<unsigned long long>(MNN::VulkanArmNx::recordedDispatchCount()));
    return 0;
}
