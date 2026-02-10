//
//  lookahead.hpp
//
//  Created by MNN on 2025/04/09.
//

#ifndef LOOK_AHEAD_HPP
#define LOOK_AHEAD_HPP

namespace MNN {
namespace Transformer {

enum class MatchStrictLevel : int {
    // adopt draft as long as ngram matched
    LOW_LEVEL,
    // adopt draft not only ngram matched, but also check ngram match size and concentration degree
    MEDIUM_LEVEL,
    // adopt draft not only ngram matched, but also check ngram match size and concentration degree strictly
    HIGH_LEVEL,
};
enum class NgramSelectRule : int {
    // match length * match frequency -> score
    FreqxLen_RULE,
    // first come first serve
    FCFS_RULE,
};

} // namespace Transformer
} // namespace MNN
#endif
