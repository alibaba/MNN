#ifndef OMNI_AUDIO_UTILS_HPP
#define OMNI_AUDIO_UTILS_HPP

#include <vector>

namespace MNN {
namespace Transformer {

std::vector<int> buildOmniAudioWindowBoundaries(int seqlen, int n_window);

} // namespace Transformer
} // namespace MNN

#endif // OMNI_AUDIO_UTILS_HPP
