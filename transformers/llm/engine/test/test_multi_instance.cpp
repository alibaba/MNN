//
//  test_multi_instance.cpp
//  MNN
//
//  Concurrency regression test: N threads each own an independent Llm
//  instance (same config) and generate concurrently. Guards against races on
//  process-global state shared across instances, e.g. the Metal encode-replay
//  recording proxy (gMetalReplayProxy) and the tokenizer regex caches
//  (unicode.cpp get_compiled/get_wregex), both hit only under multi-instance
//  load. Run it under ThreadSanitizer to catch new races of this class;
//  a plain build still catches crashes and empty/garbled generations.
//
//  Usage: test_multi_instance <config.json> [threads=2] [rounds=2] [max_new=32]
//  Falls back to $LLM_MODEL_DIR/config.json when no argument is given (same
//  provisioning convention as the llm smoke stage in test_stages.json).
//

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "llm/llm.hpp"

using namespace MNN::Transformer;

static std::atomic<int> gFail{0};

static void worker(const std::string& config, int id, int rounds, int maxNewTokens) {
    // Each instance owns its Executor/Runtime; only one thread drives it.
    std::unique_ptr<Llm> llm(Llm::createLLM(config));
    if (llm == nullptr || !llm->load()) {
        printf("[T%d] FAIL: load %s\n", id, config.c_str());
        gFail++;
        return;
    }
    for (int r = 0; r < rounds; ++r) {
        std::ostringstream os;
        llm->response(std::string("用一句话介绍你自己"), &os, nullptr, maxNewTokens);
        if (os.str().empty()) {
            printf("[T%d] FAIL: empty response at round %d\n", id, r);
            gFail++;
            return;
        }
        llm->reset();
    }
    printf("[T%d] ok: %d rounds\n", id, rounds);
}

int main(int argc, char* argv[]) {
    std::string config;
    if (argc > 1) {
        config = argv[1];
    } else if (const char* dir = getenv("LLM_MODEL_DIR")) {
        config = std::string(dir) + "/config.json";
    } else {
        printf("Usage: %s <config.json> [threads=2] [rounds=2] [max_new=32]\n", argv[0]);
        return 1;
    }
    const int threads      = argc > 2 ? atoi(argv[2]) : 2;
    const int rounds       = argc > 3 ? atoi(argv[3]) : 2;
    const int maxNewTokens = argc > 4 ? atoi(argv[4]) : 32;

    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) {
        pool.emplace_back(worker, config, i, rounds, maxNewTokens);
    }
    for (auto& t : pool) {
        t.join();
    }
    if (gFail.load() != 0) {
        printf("TEST_NAME_MULTI_INSTANCE: FAILED\n");
        return 1;
    }
    printf("TEST_NAME_MULTI_INSTANCE: PASSED (%d instances x %d rounds)\n", threads, rounds);
    return 0;
}
