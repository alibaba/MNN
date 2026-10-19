import Foundation

public struct LogitsProcessor {
    public var logitsWarpers: [any LogitsWarper]

    public init(logitsWarpers: [any LogitsWarper]) {
        self.logitsWarpers = logitsWarpers
    }

    public func callAsFunction(_ arr: [Float]) -> (indices: [Int], logits: [Float]) {
        var indices = Array(arr.indices)
        var logits = arr
        for warper in logitsWarpers {
            (indices, logits) = warper(indices, logits)
        }
        return (indices: indices, logits: logits)
    }
}
