//
//  omni_audio_utils.cpp
//  MNN
//
//  Created by MNN on 2026/05/21.
//

#include "omni_audio_utils.hpp"
#include <algorithm>

namespace MNN {
namespace Transformer {

std::vector<int> buildOmniAudioWindowBoundaries(int seqlen, int n_window) {
    const int clampedSeqlen = std::max(seqlen, 0);
    std::vector<int> boundaries(1, 0);
    if (n_window <= 0) {
        if (clampedSeqlen > 0) {
            boundaries.push_back(clampedSeqlen);
        }
        return boundaries;
    }
    for (int curseq = n_window; curseq < clampedSeqlen; curseq += n_window) {
        boundaries.push_back(curseq);
    }
    if (boundaries.back() != clampedSeqlen) {
        boundaries.push_back(clampedSeqlen);
    }
    return boundaries;
}

} // namespace Transformer
} // namespace MNN
