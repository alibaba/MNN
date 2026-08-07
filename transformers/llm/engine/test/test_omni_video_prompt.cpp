#include <cassert>
#include <cstdint>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include "omni.hpp"

int main() {
    std::regex multimodalRegex(MNN::Transformer::kOmniMultimodalRegex);
    std::string prompt = "describe <video>demo.mp4</video> now";
    std::smatch match;

    assert(std::regex_search(prompt, match, multimodalRegex));
    assert(match[1].str() == "video");
    assert(match[2].str() == "demo.mp4");

    prompt = "describe <img>demo.jpg</img> and <audio>demo.wav</audio>";
    assert(std::regex_search(prompt, match, multimodalRegex));
    assert(match[1].str() == "img");
    assert(match[2].str() == "demo.jpg");

    MNN::Transformer::MropeInfo pos;
    pos.push_back(1, 4, 2);
    assert(pos.currentIdx() == 5);

    std::vector<int> indices = MNN::Transformer::qwenVideoSampleIndices(25, 25.0, 2.0f, 4, 768);
    assert((indices == std::vector<int>{0, 8, 16, 24}));

    indices = MNN::Transformer::qwenVideoSampleIndices(3, 25.0, 2.0f, 4, 768);
    assert((indices == std::vector<int>{0, 1, 1, 2}));

    indices = MNN::Transformer::qwenVideoSampleIndices(100, 10.0, 10.0f, 4, 5);
    assert(indices.size() == 4);

    indices = MNN::Transformer::qwenVideoSampleIndices(1, 25.0, 2.0f, 4, 768);
    assert((indices == std::vector<int>{0, 0}));

    indices = MNN::Transformer::qwenVideoSampleIndices(1, 25.0, 2.0f, 4, 1);
    assert(indices.empty());

    assert(MNN::Transformer::qwenVideoAlignedFrameCount(5, 5, 2) == 4);
    assert(MNN::Transformer::qwenVideoAlignedFrameCount(1, 768, 2) == 2);
    assert(MNN::Transformer::qwenVideoAlignedFrameCount(1, 1, 2) == 0);

    auto videoSize = MNN::Transformer::qwenVideoResizeSize(3840, 2160, 32, 768 * 28 * 28);
    assert(videoSize.first % 32 == 0);
    assert(videoSize.second % 32 == 0);
    assert(static_cast<int64_t>(videoSize.first) * videoSize.second <= 768 * 28 * 28);
    videoSize = MNN::Transformer::qwenVideoResizeSize(854, 854, 28, 0);
    assert((videoSize == std::make_pair(840, 840)));

    int frameCount = MNN::Transformer::qwenVideoAlignedFrameCount(20, 768, 2);
    int maxPixels = MNN::Transformer::qwenVideoEffectiveMaxPixels(768 * 28 * 28, 4096, frameCount, 2, 16);
    videoSize = MNN::Transformer::qwenVideoResizeSize(3840, 2160, 32, maxPixels);
    int seqLen = frameCount / 2 * (videoSize.second / 16) * (videoSize.first / 16);
    assert(seqLen <= 4096);
    assert(MNN::Transformer::qwenVideoEffectiveMaxPixels(0, 4096, frameCount, 2, 16) == 409 * 16 * 16);

    std::vector<float> mask(16);
    MNN::Transformer::fillQwenVisionAttentionMask(mask.data(), 2, 2);
    const float blocked = std::numeric_limits<float>::lowest();
    assert(mask[0] == 0.0f);
    assert(mask[1] == 0.0f);
    assert(mask[2] == blocked);
    assert(mask[3] == blocked);
    assert(mask[8] == blocked);
    assert(mask[9] == blocked);
    assert(mask[10] == 0.0f);
    assert(mask[11] == 0.0f);
    return 0;
}
