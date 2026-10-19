import Foundation

public struct TemperatureLogitsWarper: LogitsWarper {
    public var temperature: Float

    public init(temperature: Float) {
        self.temperature = temperature
    }

    public func warp(indices: [Int], logits: [Float]) -> (indices: [Int], logits: [Float]) {
        (indices: indices, logits: logits.map { $0 / temperature })
    }
}
