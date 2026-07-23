#include <cassert>
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
    assert((indices == std::vector<int>{0, 1, 2}));

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
