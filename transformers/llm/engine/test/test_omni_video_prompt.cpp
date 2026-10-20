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
static int gChecks   = 0;

#define CHECK_TRUE(...)                                       \
    do {                                                      \
        ++gChecks;                                            \
        if (!(__VA_ARGS__)) {                                 \
            ++gFailures;                                      \
            printf("FAILED line %d: %s\n", __LINE__, #__VA_ARGS__); \
        }                                                     \
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

    std::vector<int> indices = MNN::Transformer::qwenVideoSampleIndices(25, 25.0, 2.0f, 4, 768);
    CHECK_TRUE(indices == std::vector<int>({0, 8, 16, 24}));

    indices = MNN::Transformer::qwenVideoSampleIndices(3, 25.0, 2.0f, 4, 768);
    CHECK_TRUE(indices == std::vector<int>({0, 1, 1, 2}));

    indices = MNN::Transformer::qwenVideoSampleIndices(100, 10.0, 10.0f, 4, 5);
    CHECK_TRUE(indices.size() == 4);

    indices = MNN::Transformer::qwenVideoSampleIndices(1, 25.0, 2.0f, 4, 768);
    CHECK_TRUE(indices == std::vector<int>({0, 0}));

    indices = MNN::Transformer::qwenVideoSampleIndices(1, 25.0, 2.0f, 4, 1);
    CHECK_TRUE(indices.empty());

    CHECK_TRUE(MNN::Transformer::qwenVideoAlignedFrameCount(5, 5, 2) == 4);
    CHECK_TRUE(MNN::Transformer::qwenVideoAlignedFrameCount(1, 768, 2) == 2);
    CHECK_TRUE(MNN::Transformer::qwenVideoAlignedFrameCount(1, 1, 2) == 0);

    auto videoSize = MNN::Transformer::qwenVideoResizeSize(3840, 2160, 32, 768 * 28 * 28);
    CHECK_TRUE(videoSize.first % 32 == 0);
    CHECK_TRUE(videoSize.second % 32 == 0);
    CHECK_TRUE(static_cast<int64_t>(videoSize.first) * videoSize.second <= 768 * 28 * 28);
    videoSize = MNN::Transformer::qwenVideoResizeSize(854, 854, 28, 0);
    CHECK_TRUE(videoSize == std::make_pair(840, 840));

    int frameCount = MNN::Transformer::qwenVideoAlignedFrameCount(20, 768, 2);
    int maxPixels = MNN::Transformer::qwenVideoEffectiveMaxPixels(768 * 28 * 28, 4096, frameCount, 2, 16);
    videoSize = MNN::Transformer::qwenVideoResizeSize(3840, 2160, 32, maxPixels);
    int seqLen = frameCount / 2 * (videoSize.second / 16) * (videoSize.first / 16);
    CHECK_TRUE(seqLen <= 4096);
    CHECK_TRUE(MNN::Transformer::qwenVideoEffectiveMaxPixels(0, 4096, frameCount, 2, 16) == 409 * 16 * 16);

    // Per-frame temporal blocks: with gridT=2 and 2 tokens per frame the 4x4 plane is two
    // 2x2 diagonal blocks, everything across frames masked. This is the fill that
    // fillBlockDiagonal + uniformBoundaries now produce (llm/attention_mask.hpp).
    std::vector<float> mask(16);
    MNN::Transformer::fillQwenVisionAttentionMask(mask.data(), 2, 2);
    const float blocked = std::numeric_limits<float>::lowest();
    CHECK_TRUE(mask[0] == 0.0f);
    CHECK_TRUE(mask[1] == 0.0f);
    CHECK_TRUE(mask[2] == blocked);
    CHECK_TRUE(mask[3] == blocked);
    CHECK_TRUE(mask[8] == blocked);
    CHECK_TRUE(mask[9] == blocked);
    CHECK_TRUE(mask[10] == 0.0f);
    CHECK_TRUE(mask[11] == 0.0f);

    printf("test_omni_video_prompt: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
