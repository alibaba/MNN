import Foundation

/// `RepetitionPenaltyWarper` prevents the repetition of previous tokens through a penalty.
/// This penalty is applied at most once per token.
/// https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L294
public struct RepetitionPenaltyWarper: LogitsWarper {
    public var penalty: Float

    public init(penalty: Double) {
        self.penalty = Float(penalty)
    }

    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        var logits = logits
        for index in indices {
            if logits[index] < 0 {
                logits[index] *= penalty
            } else {
                logits[index] /= penalty
            }
        }

        return (indices, logits)
    }
}
