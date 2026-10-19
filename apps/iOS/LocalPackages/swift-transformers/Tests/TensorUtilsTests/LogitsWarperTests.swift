//
//  LogitsWarperTests.swift
//
//  Created by Jan Krukowski on 09/12/2023.
//

import CoreML
@testable import TensorUtils
import XCTest

final class LogitsWarperTests: XCTestCase {
    private let accuracy: Float = 0.00001

    func testTemperatureLogitsWarper() {
        let result1 = TemperatureLogitsWarper(temperature: 0.0)([], [])
        XCTAssertTrue(result1.indices.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let result2 = TemperatureLogitsWarper(temperature: 1.0)([], [])
        XCTAssertTrue(result2.indices.isEmpty)
        XCTAssertTrue(result2.logits.isEmpty)

        let result3 = TemperatureLogitsWarper(temperature: 1.0)([0, 1], [2.0, 1.0])
        XCTAssertEqual(result3.indices, [0, 1])
        XCTAssertEqual(result3.logits, [2.0, 1.0], accuracy: accuracy)

        let result4 = TemperatureLogitsWarper(temperature: 2.0)([0, 1], [2.0, 1.0])
        XCTAssertEqual(result4.indices, [0, 1])
        XCTAssertEqual(result4.logits, [1.0, 0.5], accuracy: accuracy)

        let result5 = TemperatureLogitsWarper(temperature: 0.5)([0, 1], [2.0, 1.0])
        XCTAssertEqual(result5.indices, [0, 1])
        XCTAssertEqual(result5.logits, [4.0, 2.0], accuracy: accuracy)

        let result6 = TemperatureLogitsWarper(temperature: 0.5)([200, 100], [2.0, 1.0])
        XCTAssertEqual(result6.indices, [200, 100])
        XCTAssertEqual(result6.logits, [4.0, 2.0], accuracy: accuracy)
    }

    func testTopKLogitsWarper() {
        let result1 = TopKLogitsWarper(k: 0)([], [])
        XCTAssertTrue(result1.indices.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let result2 = TopKLogitsWarper(k: 3)([], [])
        XCTAssertTrue(result2.indices.isEmpty)
        XCTAssertTrue(result2.logits.isEmpty)

        let result3 = TopKLogitsWarper(k: 3)([0, 1], [2.0, 1.0])
        XCTAssertEqual(result3.indices, [0, 1])
        XCTAssertEqual(result3.logits, [2.0, 1.0], accuracy: accuracy)

        let result4 = TopKLogitsWarper(k: 3)([0, 1, 2], [2.0, 1.0, 3.0])
        XCTAssertEqual(result4.indices, [2, 0, 1])
        XCTAssertEqual(result4.logits, [3.0, 2.0, 1.0], accuracy: accuracy)

        let result5 = TopKLogitsWarper(k: 4)([0, 1, 2, 3, 4, 5], [2.0, 1.0, 3.0, -1.0, 123.0, 0.0])
        XCTAssertEqual(result5.indices, [4, 2, 0, 1])
        XCTAssertEqual(result5.logits, [123.0, 3.0, 2.0, 1.0], accuracy: accuracy)

        let result6 = TopKLogitsWarper(k: 3)([10, 1, 52], [2.0, 1.0, 3.0])
        XCTAssertEqual(result6.indices, [52, 10, 1])
        XCTAssertEqual(result6.logits, [3.0, 2.0, 1.0], accuracy: accuracy)
    }

    func testTopPLogitsWarper() {
        let result1 = TopPLogitsWarper(p: 0.99)([], [])
        XCTAssertTrue(result1.indices.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let logits = (0..<10).map { Float($0) }
        let indices = Array(logits.indices)
        let result2 = TopPLogitsWarper(p: 0.99)(indices, logits)
        XCTAssertEqual(result2.indices, [9, 8, 7, 6, 5])
        XCTAssertEqual(result2.logits, [9.0, 8.0, 7.0, 6.0, 5.0], accuracy: accuracy)

        let result3 = TopPLogitsWarper(p: 0.95)(indices, logits)
        XCTAssertEqual(result3.indices, [9, 8, 7])
        XCTAssertEqual(result3.logits, [9.0, 8.0, 7.0], accuracy: accuracy)

        let result4 = TopPLogitsWarper(p: 0.6321493)(indices, logits)
        XCTAssertEqual(result4.indices, [9, 8])
        XCTAssertEqual(result4.logits, [9.0, 8.0], accuracy: accuracy)

        let result5 = TopPLogitsWarper(p: 0.95)([3, 1, 8], [0, 1, 2])
        XCTAssertEqual(result5.indices, [8, 1, 3])
        XCTAssertEqual(result5.logits, [2, 1, 0], accuracy: accuracy)
    }

    func testRepetitionPenaltyWarper() {
        let indices = Array(0..<10)
        let logits = indices.map { Float($0) }

        let result1 = RepetitionPenaltyWarper(penalty: 1.0)(indices, logits)
        XCTAssertEqual(result1.indices, indices)
        XCTAssertEqual(result1.logits, logits, accuracy: accuracy)

        let result2 = RepetitionPenaltyWarper(penalty: 3.75)(indices, logits)
        XCTAssertEqual(result2.indices, indices)
        let logits2 = indices.map { Float($0) / 3.75 }
        XCTAssertEqual(result2.logits, logits2, accuracy: accuracy)

        let result3 = RepetitionPenaltyWarper(penalty: 0.75)([0, 1, 2], [0.8108, 0.9954, 0.0119])
        XCTAssertEqual(result3.indices, [0, 1, 2])
        XCTAssertEqual(result3.logits, [1.0811, 1.3272, 0.0158], accuracy: 1e-4)

        let result4 = RepetitionPenaltyWarper(penalty: 1.11)([2, 3, 4], [0.5029, 0.8694, 0.4765, 0.9967, 0.4190, 0.9158])
        XCTAssertEqual(result4.indices, [2, 3, 4])
        XCTAssertEqual(result4.logits, [0.5029, 0.8694, 0.4293, 0.8980, 0.3775, 0.9158], accuracy: 1e-4)

        let result5 = RepetitionPenaltyWarper(penalty: 0.9)([0, 1, 2], [-0.7433, -0.4738, -0.2966])
        XCTAssertEqual(result5.indices, [0, 1, 2])
        XCTAssertEqual(result5.logits, [-0.6690, -0.4264, -0.2669], accuracy: 1e-4)

        let result6 = RepetitionPenaltyWarper(penalty: 1.125)([3, 1, 2], [0.1674, 0.6431, 0.6780, 0.2755])
        XCTAssertEqual(result6.indices, [3, 1, 2])
        XCTAssertEqual(result6.logits, [0.1674, 0.5716, 0.6026, 0.2449], accuracy: 1e-4)
    }

    func testLogitsProcessor() {
        let processor1 = LogitsProcessor(logitsWarpers: [])
        let result1 = processor1([])
        XCTAssertTrue(result1.indices.isEmpty)
        XCTAssertTrue(result1.logits.isEmpty)

        let processor2 = LogitsProcessor(logitsWarpers: [])
        let result2 = processor2([2.0, 1.0])
        XCTAssertEqual(result2.indices, [0, 1])
        XCTAssertEqual(result2.logits, [2.0, 1.0], accuracy: accuracy)

        let processor3 = LogitsProcessor(
            logitsWarpers: [TopKLogitsWarper(k: 3)]
        )
        let result3 = processor3([2.0, 1.0, 3.0, -5.0])
        XCTAssertEqual(result3.indices, [2, 0, 1])
        XCTAssertEqual(result3.logits, [3.0, 2.0, 1.0], accuracy: accuracy)

        let processor4 = LogitsProcessor(
            logitsWarpers: [TopKLogitsWarper(k: 3), TopPLogitsWarper(p: 0.99)]
        )
        let result4 = processor4([2.0, 1.0, 3.0, -5.0, -23.0, 12.5])
        XCTAssertEqual(result4.indices, [5])
        XCTAssertEqual(result4.logits, [12.5], accuracy: accuracy)

        let processor5 = LogitsProcessor(
            logitsWarpers: [TopKLogitsWarper(k: 4), TopPLogitsWarper(p: 0.99)]
        )
        let result5 = processor5([2.0, 1.0, 3.0, -5.0, -3.0, 4.5])
        XCTAssertEqual(result5.indices, [5, 2, 0, 1])
        XCTAssertEqual(result5.logits, [4.5, 3.0, 2.0, 1.0], accuracy: accuracy)
    }
}
