import Foundation

/// Protocol for all logit warpers that can be applied during generation
public protocol LogitsWarper {
    func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float])
    func callAsFunction(_ indices: [Int], _ logits: [Float]) -> (indices: [Int], logits: [Float])
}

public extension LogitsWarper {
    func callAsFunction(_ indices: [Int], _ logits: [Float]) -> (indices: [Int], logits: [Float]) {
        warp(indices: indices, logits: logits)
    }
}
