//
//  multi_lora_demo.cpp
//
//  Verify that two split LoRA models can stay loaded and run concurrently,
//  then be selected repeatedly without state leaking between adapters.
//

#include "llm/llm.hpp"

#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace MNN::Transformer;

struct RunResult {
    std::string name;
    std::string expected;
    std::string generated;
    bool success = false;
};

static RunResult runAdapter(const std::string& name, Llm* llm, const std::string& prompt, const std::string& expected,
                            const std::string& otherExpected) {
    llm->reset();
    std::ostringstream output;
    llm->response(prompt, &output, nullptr, 16);

    RunResult result;
    result.name = name;
    result.expected = expected;
    result.generated = output.str();
    const bool hasExpected = result.generated.find(expected) != std::string::npos;
    const bool hasOther = !otherExpected.empty() && result.generated.find(otherExpected) != std::string::npos;
    const bool runtimeOk = llm->getContext()->status != LlmStatus::INTERNAL_ERROR;
    result.success = runtimeOk && hasExpected && !hasOther;
    return result;
}

static void printResult(const std::string& phase, const RunResult& result) {
    std::cout << "[" << phase << "] " << result.name << ": " << (result.success ? "PASS" : "FAIL") << "\n"
              << "  expected: " << result.expected << "\n"
              << "  generated: " << result.generated << std::endl;
}

static std::unique_ptr<Llm> createAdapter(Llm* base, const std::string& loraPath, const std::string& name) {
    std::unique_ptr<Llm> adapter(base->create_lora(loraPath));
    if (adapter == nullptr) {
        std::cerr << "Failed to load " << name << " from: " << loraPath << std::endl;
        return nullptr;
    }
    adapter->set_config(R"({"async":false,"temperature":0,"top_k":1,"top_p":1.0,"max_new_tokens":16})");
    return adapter;
}

int main(int argc, const char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " CONFIG LORA_A EXPECTED_A LORA_B EXPECTED_B [PROMPT] [ROUNDS]"
                  << std::endl;
        return 2;
    }

    const std::string configPath = argv[1];
    const std::string loraAPath = argv[2];
    const std::string expectedA = argv[3];
    const std::string loraBPath = argv[4];
    const std::string expectedB = argv[5];
    const std::string prompt = argc >= 7 ? argv[6] : "适配器切换测试：请只输出当前适配器口令。";
    int rounds = argc >= 8 ? std::atoi(argv[7]) : 2;
    if (rounds <= 0) {
        std::cerr << "ROUNDS must be greater than zero." << std::endl;
        return 2;
    }

    // The base must outlive all adapters because create_lora() shares its base
    // module. Declaration order guarantees adapters are destroyed first.
    std::unique_ptr<Llm> base(Llm::createLLM(configPath));
    if (base == nullptr || !base->load()) {
        std::cerr << "Failed to load base model from: " << configPath << std::endl;
        return 1;
    }

    std::unique_ptr<Llm> adapterA = createAdapter(base.get(), loraAPath, "adapter A");
    std::unique_ptr<Llm> adapterB = createAdapter(base.get(), loraBPath, "adapter B");
    if (adapterA == nullptr || adapterB == nullptr) {
        return 1;
    }

    bool allPassed = true;
    auto futureA =
        std::async(std::launch::async, runAdapter, "adapter A", adapterA.get(), prompt, expectedA, expectedB);
    auto futureB =
        std::async(std::launch::async, runAdapter, "adapter B", adapterB.get(), prompt, expectedB, expectedA);
    const RunResult parallelA = futureA.get();
    const RunResult parallelB = futureB.get();
    printResult("parallel", parallelA);
    printResult("parallel", parallelB);
    allPassed = parallelA.success && parallelB.success;

    for (int round = 0; round < rounds; ++round) {
        const RunResult switchedA = runAdapter("adapter A", adapterA.get(), prompt, expectedA, expectedB);
        const RunResult switchedB = runAdapter("adapter B", adapterB.get(), prompt, expectedB, expectedA);
        printResult("switch " + std::to_string(round + 1), switchedA);
        printResult("switch " + std::to_string(round + 1), switchedB);
        allPassed = allPassed && switchedA.success && switchedB.success;
    }

    std::cout << (allPassed ? "MULTI_LORA_TEST_PASS" : "MULTI_LORA_TEST_FAIL") << std::endl;
    return allPassed ? 0 : 1;
}
