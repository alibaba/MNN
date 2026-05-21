//
//  test_omni_audio_window.cpp
//  MNN
//

#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "omni_audio_utils.hpp"

static bool expectBoundaries(int seqlen, int n_window, const std::vector<int>& expected) {
    const auto actual = MNN::Transformer::buildOmniAudioWindowBoundaries(seqlen, n_window);
    if (actual == expected) {
        return true;
    }
    std::cerr << "Unexpected boundaries for seqlen=" << seqlen << ", n_window=" << n_window << std::endl;
    std::cerr << "Expected:";
    for (int value : expected) {
        std::cerr << ' ' << value;
    }
    std::cerr << std::endl;
    std::cerr << "Actual:";
    for (int value : actual) {
        std::cerr << ' ' << value;
    }
    std::cerr << std::endl;
    return false;
}

static bool expectMaskLayout() {
    const int seqlen = 200;
    std::vector<float> mask(seqlen * seqlen, std::numeric_limits<float>::lowest());
    const auto boundaries = MNN::Transformer::buildOmniAudioWindowBoundaries(seqlen, 100);
    for (size_t i = 1; i < boundaries.size(); ++i) {
        for (int row = boundaries[i - 1]; row < boundaries[i]; ++row) {
            for (int col = boundaries[i - 1]; col < boundaries[i]; ++col) {
                mask[seqlen * row + col] = 0.0f;
            }
        }
    }
    if (mask[0] != 0.0f || mask[99 * seqlen + 99] != 0.0f ||
        mask[100 * seqlen + 100] != 0.0f || mask[199 * seqlen + 199] != 0.0f ||
        mask[99 * seqlen + 100] != std::numeric_limits<float>::lowest()) {
        std::cerr << "Unexpected mask layout for exact window boundaries" << std::endl;
        return false;
    }
    return true;
}

int main() {
    bool ok = true;
    ok &= expectBoundaries(100, 100, {0, 100});
    ok &= expectBoundaries(101, 100, {0, 100, 101});
    ok &= expectBoundaries(200, 100, {0, 100, 200});
    ok &= expectBoundaries(99, 100, {0, 99});
    ok &= expectMaskLayout();
    return ok ? 0 : 1;
}
