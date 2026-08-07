import Accelerate
import Foundation

/// Top-K.
/// Select the k most-probable element indices from `arr`
/// and return both the indices (from the original array)
/// and their probabilities.
public struct TopKLogitsWarper: LogitsWarper {
    public var k: Int

    public init(k: Int) {
        self.k = k
    }

    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        guard !logits.isEmpty else {
            return (indices: [], logits: [])
        }
        let k = min(k, logits.count)
        let arrDescriptor = BNNSNDArrayDescriptor.allocate(
            initializingFrom: logits,
            shape: .vector(logits.count)
        )
        defer {
            arrDescriptor.deallocate()
        }
        let bestIndices = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Int32.self,
            shape: .vector(k)
        )
        defer {
            bestIndices.deallocate()
        }
        let bestValues = BNNSNDArrayDescriptor.allocateUninitialized(
            scalarType: Float.self,
            shape: .vector(k)
        )
        defer {
            bestValues.deallocate()
        }
        try! Accelerate.BNNS.applyTopK(
            k: k,
            input: arrDescriptor,
            bestValues: bestValues,
            bestIndices: bestIndices,
            axis: 0,
            batchSize: 1,
            filterParameters: nil
        )
        let topkLogits = bestValues.data!.withMemoryRebound(to: Float.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        let topkIndices = bestIndices.data!.withMemoryRebound(to: Int32.self, capacity: k) { ptr in
            Array(UnsafeBufferPointer(start: ptr, count: k))
        }
        return (indices: topkIndices.map { indices[Int($0)] }, logits: topkLogits)
    }
}
