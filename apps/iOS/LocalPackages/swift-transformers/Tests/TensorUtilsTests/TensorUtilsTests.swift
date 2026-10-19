//
//  TensorUtilsTests.swift
//
//  Created by Jan Krukowski on 25/11/2023.
//

import CoreML
@testable import TensorUtils
import XCTest

final class TensorUtilsTests: XCTestCase {
    private let accuracy: Float = 0.00001

    func testCumsum() {
        XCTAssertTrue(Math.cumsum([]).isEmpty)
        XCTAssertEqual(Math.cumsum([1]), [1])
        XCTAssertEqual(Math.cumsum([1, 2, 3, 4]), [1, 3, 6, 10])
    }

    func testArgMax() throws {
        let result1 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Float], count: 4)
        XCTAssertEqual(result1.0, 1)
        XCTAssertEqual(result1.1, 4.0)

        let result2 = Math.argmax32([3.0, 4.0, 1.0, 2.0], count: 4)
        XCTAssertEqual(result2.0, 1)
        XCTAssertEqual(result2.1, 4.0)

        let result3 = Math.argmax([3.0, 4.0, 1.0, 2.0] as [Double], count: 4)
        XCTAssertEqual(result3.0, 1)
        XCTAssertEqual(result3.1, 4.0)

        let result4 = try Math.argmax32(MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Float]))
        XCTAssertEqual(result4.0, 1)
        XCTAssertEqual(result4.1, 4.0)

        let result5 = try Math.argmax(MLMultiArray([3.0, 4.0, 1.0, 2.0] as [Double]))
        XCTAssertEqual(result5.0, 1)
        XCTAssertEqual(result5.1, 4.0)

        let result6 = Math.argmax(MLShapedArray(scalars: [3.0, 4.0, 1.0, 2.0] as [Float], shape: [4]))
        XCTAssertEqual(result6.0, 1)
        XCTAssertEqual(result6.1, 4.0)
    }

    func testSoftmax() {
        XCTAssertEqual(Math.softmax([]), [])

        let result1 = Math.softmax([3.0, 4.0, 1.0, 2.0])
        XCTAssertEqual(result1, [0.23688284, 0.6439143, 0.032058604, 0.08714432], accuracy: accuracy)
        XCTAssertEqual(result1.reduce(0, +), 1.0, accuracy: accuracy)
    }
}
