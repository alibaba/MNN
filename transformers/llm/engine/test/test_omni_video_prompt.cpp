#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include "omni.hpp"

// assert() is compiled out under NDEBUG and CI builds Release (-DCMAKE_BUILD_TYPE=Release
// implies -DNDEBUG), so a test written with assert() passes unconditionally there. These
// checks have to survive that: count the failures, report them, and exit non-zero.
// Variadic, because conditions like `x == std::vector<int>{0, 8, 16}` contain commas that a
// single-argument macro would split.
static int gFailures = 0;
static int gChecks = 0;

#define CHECK_TRUE(...)                                             \
    do {                                                            \
        ++gChecks;                                                  \
        if (!(__VA_ARGS__)) {                                       \
            ++gFailures;                                            \
            printf("FAILED line %d: %s\n", __LINE__, #__VA_ARGS__); \
        }                                                           \
    } while (0)

int main() {
    std::regex multimodalRegex(MNN::Transformer::kOmniMultimodalRegex);
    std::string prompt = "describe <video>demo.mp4</video> now";
    std::smatch match;

    CHECK_TRUE(std::regex_search(prompt, match, multimodalRegex));
    CHECK_TRUE(match[1].str() == "video");
    CHECK_TRUE(match[2].str() == "demo.mp4");

    prompt = "describe <img>demo.jpg</img> and <audio>demo.wav</audio>";
    CHECK_TRUE(std::regex_search(prompt, match, multimodalRegex));
    CHECK_TRUE(match[1].str() == "img");
    CHECK_TRUE(match[2].str() == "demo.jpg");

    MNN::Transformer::MropeInfo pos;
    pos.push_back(1, 4, 2);
    CHECK_TRUE(pos.currentIdx() == 5);

    // Temporal groups attend within their own block.
    std::vector<float> mask(16);
    auto boundaries = MNN::Transformer::uniformBoundaries(4, 2);
    MNN::Transformer::fillBlockDiagonal(mask.data(), 4, 4, boundaries);
    const float blocked = std::numeric_limits<float>::lowest();
    CHECK_TRUE(mask[0] == 0.0f);
    CHECK_TRUE(mask[1] == 0.0f);
    CHECK_TRUE(mask[2] == blocked);
    CHECK_TRUE(mask[3] == blocked);
    CHECK_TRUE(mask[8] == blocked);
    CHECK_TRUE(mask[9] == blocked);
    CHECK_TRUE(mask[10] == 0.0f);
    CHECK_TRUE(mask[11] == 0.0f);

    pos.prependTextPositions(2);
    CHECK_TRUE(pos.mT == std::vector<int>({0, 1, 3}));
    CHECK_TRUE(pos.mH == std::vector<int>({0, 1, 6}));
    CHECK_TRUE(pos.mW == std::vector<int>({0, 1, 4}));
    CHECK_TRUE(pos.currentIdx() == 7);
    const auto copy = pos;
    pos.clear();
    CHECK_TRUE(pos.currentIdx() == 0);
    CHECK_TRUE(copy.currentIdx() == 7);

    printf("test_omni_video_prompt: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
