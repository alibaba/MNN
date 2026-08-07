import Foundation

/// Top-P.
/// Select the smallest set of elements whose cumulative probability exceeds the probability `p`.
/// Based on https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
public struct TopPLogitsWarper: LogitsWarper {
    public var p: Float

    public init(p: Float) {
        self.p = p
    }

    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard !logits.isEmpty else {
            return (indices: [], logits: [])
        }

        let arrSoftmax = Math.softmax(logits)
        var indexLogitProb = [(index: Int, logit: Float, prob: Float)]()
        indexLogitProb.reserveCapacity(logits.count)
        for (index, data) in zip(logits, arrSoftmax).enumerated() {
            indexLogitProb.append((index: index, logit: data.0, prob: data.1))
        }
        indexLogitProb.sort { $0.prob > $1.prob }

        let cumsum = Math.cumsum(indexLogitProb.map(\.prob))
        var sliceIndex = cumsum.count - 1
        for (index, element) in cumsum.enumerated() where element > p {
            sliceIndex = index
            break
        }

        let toppIndices = indexLogitProb[0...sliceIndex].map { indices[$0.index] }
        let toppLogits = indexLogitProb[0...sliceIndex].map(\.logit)
        return (indices: toppIndices, logits: toppLogits)
    }
}
