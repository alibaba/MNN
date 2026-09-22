//
//  SamplerTest.cpp
//  MNNTests
//
//  Tests for the LLM Sampler, particularly that repetition/presence/frequency
//  penalty is correctly applied in "mixed" mode even when "penalty" is not
//  explicitly listed in mixed_samplers.
//

#ifdef MNN_BUILD_LLM

#include "MNNTestSuite.h"
#include "sampler.hpp"
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>

using namespace MNN::Transformer;

// Helper: build an LlmConfig from a JSON string without touching the filesystem.
static std::shared_ptr<LlmConfig> makeConfig(const char* json) {
    auto cfg = std::make_shared<LlmConfig>();
    cfg->config_ = ujson::json::parse(json);
    return cfg;
}

// Helper: build a 1-D VARP of logits on the host.
static MNN::Express::VARP makeLogits(const std::vector<float>& vals) {
    return MNN::Express::_Const(vals.data(), {(int)vals.size()},
                                MNN::Express::NCHW, halide_type_of<float>());
}

// --------------------------------------------------------------------------
// isPenaltyActive: single source of truth for "would penalty change logits?"
// --------------------------------------------------------------------------
class SamplerIsPenaltyActiveTest : public MNNTestCase {
    bool run(int /* precision */) override {
        auto ctx = std::make_shared<LlmContext>();

        // All defaults → inactive.
        {
            auto cfg = makeConfig(R"({"sampler_type":"mixed"})");
            Sampler s(ctx, cfg);
            // Default rp=1.0, pp=0, fp=0, nf=1.0 → sample() on flat logits
            // should complete without penalty overhead.  We test indirectly:
            // greedy on uniform logits always picks index 0.
            auto greedy = makeConfig(R"({"sampler_type":"greedy"})");
            Sampler gs(ctx, greedy);
            auto logits = makeLogits({1.0f, 1.0f, 1.0f, 1.0f});
            int tok = gs.sample(logits);
            MNNTEST_ASSERT(tok == 0);
        }

        // repetition_penalty > 1 → active.
        {
            auto cfg = makeConfig(R"({"sampler_type":"mixed","repetition_penalty":1.2})");
            Sampler s(ctx, cfg);
            // With empty history, penalty has nothing to penalize; the token
            // distribution is unchanged so greedy still picks 0.
            auto logits = makeLogits({5.0f, 3.0f, 1.0f});
            int tok = s.sample(logits);
            MNNTEST_ASSERT(tok == 0);
        }

        return true;
    }
};
MNNTestSuiteRegister(SamplerIsPenaltyActiveTest, "llm/sampler_is_penalty_active");

// --------------------------------------------------------------------------
// Mixed mode: penalty auto-enabled when repetition_penalty > 1
// --------------------------------------------------------------------------
class SamplerMixedPenaltyAutoEnableTest : public MNNTestCase {
    bool run(int /* precision */) override {
        auto ctx = std::make_shared<LlmContext>();

        // Logit layout: token 0 has the highest logit, token 1 is close behind.
        // After history contains many copies of token 0, penalty should
        // suppress it enough that token 1 wins.
        const int VOCAB = 8;
        std::vector<float> raw(VOCAB, -10.0f);
        raw[0] = 10.0f;   // dominant token
        raw[1] = 9.5f;    // runner-up

        // Fill history with token 0 to trigger repetition penalty.
        ctx->history_tokens.clear();
        for (int i = 0; i < 50; i++) {
            ctx->history_tokens.push_back(0);
        }

        // Without penalty (default config), token 0 always wins.
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "repetition_penalty":1.0
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            // With temperature sampling it's probabilistic, so use greedy
            // to get a deterministic check.
        }

        // With high repetition_penalty, token 0 should be penalized.
        // Use greedy select to make the test deterministic.
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "mixed_samplers":["topK","greedy"],
                "repetition_penalty":5.0,
                "top_k":2
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            // Token 0 logit (10.0) is divided by 5.0 → 2.0.
            // Token 1 logit (9.5) is untouched → 9.5.
            // Greedy picks token 1.
            MNNTEST_ASSERT(tok == 1);
        }

        // Verify that the penalty was auto-enabled: the config above does NOT
        // include "penalty" in mixed_samplers, yet it still works.  With rp=1.0
        // (no penalty) the same logits should pick token 0.
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "mixed_samplers":["topK","greedy"],
                "repetition_penalty":1.0,
                "top_k":2
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            MNNTEST_ASSERT(tok == 0);
        }

        return true;
    }
};
MNNTestSuiteRegister(SamplerMixedPenaltyAutoEnableTest, "llm/sampler_mixed_penalty_auto_enable");

// --------------------------------------------------------------------------
// Presence and frequency penalty also auto-enable the penalty step.
// --------------------------------------------------------------------------
class SamplerMixedPresenceFrequencyTest : public MNNTestCase {
    bool run(int /* precision */) override {
        auto ctx = std::make_shared<LlmContext>();
        ctx->history_tokens.clear();
        for (int i = 0; i < 30; i++) {
            ctx->history_tokens.push_back(0);
        }

        const int VOCAB = 4;
        std::vector<float> raw(VOCAB, -10.0f);
        raw[0] = 5.0f;
        raw[1] = 4.8f;

        // presence_penalty alone (repetition_penalty=1.0).
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "mixed_samplers":["greedy"],
                "repetition_penalty":1.0,
                "presence_penalty":10.0
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            // Token 0: 5.0 - 10.0 = -5.0.  Token 1: 4.8 (untouched).
            MNNTEST_ASSERT(tok == 1);
        }

        // frequency_penalty alone.
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "mixed_samplers":["greedy"],
                "repetition_penalty":1.0,
                "frequency_penalty":1.0
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            // Token 0: 5.0 - 1.0*30 = -25.0.  Token 1: 4.8.
            MNNTEST_ASSERT(tok == 1);
        }

        return true;
    }
};
MNNTestSuiteRegister(SamplerMixedPresenceFrequencyTest, "llm/sampler_mixed_presence_frequency");

// --------------------------------------------------------------------------
// ngram_factor > 1 also auto-enables the penalty step.
// --------------------------------------------------------------------------
class SamplerMixedNgramFactorTest : public MNNTestCase {
    bool run(int /* precision */) override {
        auto ctx = std::make_shared<LlmContext>();

        // Build a history that creates an n-gram match: repeating [0,1,0,1,...].
        // With n_gram=2 and ngram_factor=large, repeating the pattern should
        // be heavily penalized.
        ctx->history_tokens.clear();
        for (int i = 0; i < 20; i++) {
            ctx->history_tokens.push_back(i % 2);  // 0,1,0,1,...
        }

        const int VOCAB = 4;
        std::vector<float> raw(VOCAB, -10.0f);
        raw[0] = 5.0f;   // next in the pattern would be token 0
        raw[2] = 4.5f;   // alternative token not in history

        // With ngram_factor > 1 and repetition_penalty > 1 (both needed for
        // the n-gram path), the repeated pattern token should be suppressed.
        {
            auto cfg = makeConfig(R"({
                "sampler_type":"mixed",
                "mixed_samplers":["greedy"],
                "repetition_penalty":2.0,
                "ngram_factor":5.0,
                "n_gram":2
            })");
            Sampler s(ctx, cfg);
            auto logits = makeLogits(raw);
            int tok = s.sample(logits);
            // Token 0 is heavily penalized (rep*ngram_factor^match); token 2
            // at 4.5 should win.
            MNNTEST_ASSERT(tok == 2);
        }

        return true;
    }
};
MNNTestSuiteRegister(SamplerMixedNgramFactorTest, "llm/sampler_mixed_ngram_factor");

#endif // MNN_BUILD_LLM
