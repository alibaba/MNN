//
//  llm_logits_diff.cpp
//
//  Teacher-forced per-step logits comparison between two backends.
//  Usage: llm_logits_diff configA.json configB.json prompt.txt [max_tokens]
//  Backend A drives the trajectory (its argmax token is fed to BOTH models),
//  so every step compares logits for an identical KV/history state.
//

#include <MNN/expr/ExecutorScope.hpp>
#include <llm/llm.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;
using MNN::Express::VARP;

constexpr float NEGATIVE_INF_FLOAT = -1e30f;  // sufficiently small for float comparisons
constexpr double PROB_EPSILON = 1e-12;        // floor for probabilities before log()

struct Top2 {
    int idx0 = -1, idx1 = -1;
    float v0 = NEGATIVE_INF_FLOAT, v1 = NEGATIVE_INF_FLOAT;
};

static Top2 top2(const float* p, int n) {
    Top2 t;
    for (int i = 0; i < n; ++i) {
        float v = p[i];
        if (v > t.v0) {
            t.v1 = t.v0; t.idx1 = t.idx0;
            t.v0 = v; t.idx0 = i;
        } else if (v > t.v1) {
            t.v1 = v; t.idx1 = i;
        }
    }
    return t;
}

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s configA.json configB.json prompt.txt [max_tokens]\n", argv[0]);
        return 0;
    }
    int maxTokens = 256;
    if (argc >= 5) {
        maxTokens = atoi(argv[4]);
        if (maxTokens <= 0) {
            printf("Invalid max_tokens: %s (must be a positive integer)\n", argv[4]);
            return 1;
        }
    }

    std::unique_ptr<Llm> llmA(Llm::createLLM(argv[1]));
    std::unique_ptr<Llm> llmB(Llm::createLLM(argv[2]));
    llmA->set_config("{\"tmp_path\":\"tmp_diff_a\", \"all_logits\":false}");
    llmB->set_config("{\"tmp_path\":\"tmp_diff_b\", \"all_logits\":false}");
    if (!llmA->load() || !llmB->load()) {
        printf("load failed\n");
        return 1;
    }
    const char* noThink = R"({"jinja":{"context":{"enable_thinking":false}}})";
    llmA->set_config(noThink);
    llmB->set_config(noThink);

    std::ifstream fs(argv[3]);
    if (!fs.is_open()) {
        printf("Failed to open prompt file: %s\n", argv[3]);
        return 1;
    }
    std::stringstream ss;
    ss << fs.rdbuf();
    std::string userContent = ss.str();
    while (!userContent.empty() && (userContent.back() == '\n' || userContent.back() == '\r')) {
        userContent.pop_back();
    }
    auto prompt = llmA->apply_chat_template(userContent);
    auto ids = llmA->tokenizer_encode(prompt);
    printf("prompt tokens: %d, max new tokens: %d\n", (int)ids.size(), maxTokens);

    llmA->generate_init(nullptr, "\n");
    llmB->generate_init(nullptr, "\n");
    auto logitsA = llmA->forward(ids, true);
    auto logitsB = llmB->forward(ids, true);

    printf("step | tokA(argmaxA) tokB(argmaxB) agree | marginA marginB | maxAbsDiff meanAbsDiff | text\n");
    int disagree = 0;
    int firstDisagree = -1;
    double worstMax = 0.0;
    double sumKL = 0.0, sumNllA = 0.0, sumNllB = 0.0;
    int steps = 0;
    std::vector<double> probA, probB;
    for (int t = 0; t < maxTokens; ++t) {
        if (logitsA == nullptr || logitsB == nullptr) {
            printf("forward returned null at step %d\n", t);
            break;
        }
        int n = logitsA->getInfo()->dim.back();
        if (n <= 0) {
            printf("invalid vocab dim %d at step %d\n", n, t);
            break;
        }
        const float* pa = logitsA->readMap<float>();
        const float* pb = logitsB->readMap<float>();
        if (pa == nullptr || pb == nullptr) {
            printf("readMap returned null at step %d\n", t);
            break;
        }
        auto ta = top2(pa, n);
        auto tb = top2(pb, n);
        double maxd = 0.0, sumd = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = std::fabs((double)pa[i] - (double)pb[i]);
            if (d > maxd) maxd = d;
            sumd += d;
        }
        // softmax both, then KL(A||B) and NLL of the forced token (A's argmax)
        probA.resize(n);
        probB.resize(n);
        double za = 0.0, zb = 0.0;
        for (int i = 0; i < n; ++i) {
            probA[i] = std::exp((double)pa[i] - ta.v0);
            probB[i] = std::exp((double)pb[i] - tb.v0);
            za += probA[i];
            zb += probB[i];
        }
        double kl = 0.0;
        for (int i = 0; i < n; ++i) {
            probA[i] /= za;
            probB[i] /= zb;
            if (probA[i] > PROB_EPSILON) {
                kl += probA[i] * std::log(probA[i] / std::max(probB[i], PROB_EPSILON));
            }
        }
        sumKL += kl;
        sumNllA += -std::log(std::max(probA[ta.idx0], PROB_EPSILON));
        sumNllB += -std::log(std::max(probB[ta.idx0], PROB_EPSILON));
        ++steps;
        if (maxd > worstMax) worstMax = maxd;
        bool agree = (ta.idx0 == tb.idx0);
        if (!agree) {
            ++disagree;
            if (firstDisagree < 0) firstDisagree = t;
        }
        auto text = llmA->tokenizer_decode(ta.idx0);
        for (auto& c : text) {
            if (c == '\n') c = ' ';
        }
        printf("%4d | %6d %6d %s | %7.4f %7.4f | %9.5f %11.7f | %s\n", t, ta.idx0, tb.idx0,
               agree ? "  ==" : "DIFF", ta.v0 - ta.v1, tb.v0 - tb.v1, maxd, sumd / n, text.c_str());
        int tok = ta.idx0; // teacher forcing: A's trajectory feeds both
        if (llmA->is_stop(tok)) {
            printf("stop token at step %d\n", t);
            break;
        }
        logitsA = llmA->forward({tok}, false);
        logitsB = llmB->forward({tok}, false);
    }
    printf("\nsummary: argmax disagreements=%d, first at step %d, worst maxAbsDiff=%.5f\n", disagree,
           firstDisagree, worstMax);
    if (steps > 0) {
        printf("distribution: mean KL(A||B)=%.6f, teacher-forced mean NLL A=%.5f B=%.5f (log-ppl)\n",
               sumKL / steps, sumNllA / steps, sumNllB / steps);
    }
    return 0;
}
