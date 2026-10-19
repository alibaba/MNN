//
//  MTFontV2Tests.swift
//
//
//  Created by Peter Tang on 15/9/2023.
//

import XCTest
@testable import SwiftMath

final class MTFontV2Tests: XCTestCase {
    func testMTFontV2Script() throws {
        let size = CGFloat(Int.random(in: 20 ... 40))
        MathFont.allCases.forEach {
            let mtfont = $0.mtfont(size: size)
            let mTable = mtfont.mathTable?._mathTable
            XCTAssertNotNil(mtfont)
            XCTAssertNotNil(mTable)
        }
    }
    private let executionQueue = DispatchQueue(label: "com.swiftmath.mathbundle", attributes: .concurrent)
    private let executionGroup = DispatchGroup()
    let totalCases = 1000
    var testCount = 0
    func testConcurrentThreadsafeScript() throws {
        testCount = 0
        var mathFont: MathFont { .allCases.randomElement()! }
        for caseNumber in 0 ..< totalCases {
            helperConcurrentMTFontV2(caseNumber, mathFont: mathFont, in: executionGroup, on: executionQueue)
        }
        executionGroup.notify(queue: .main) { [weak self] in
            guard let self = self else { return }
            XCTAssertEqual(self.testCount, totalCases)
            print("\(self.testCount) completed =================")
        }
        executionGroup.wait()
    }
    func helperConcurrentMTFontV2(_ count: Int, mathFont: MathFont, in group: DispatchGroup, on queue: DispatchQueue) {
        let size = CGFloat.random(in: 20 ... 40)
        let workitem = DispatchWorkItem {
            let fontV2 = mathFont.mtfont(size: size)
            XCTAssertNotNil(fontV2)
            let (cgfont, ctfont) = (fontV2.defaultCGFont, fontV2.ctFont)
            XCTAssertNotNil(cgfont)
            XCTAssertNotNil(ctfont)
        }
        workitem.notify(queue: .main) { [weak self] in
            // print("\(Thread.isMainThread ? "main" : "global") completed .....")
            let fontV2 = mathFont.mtfont(size: size)
            XCTAssertNotNil(fontV2)
            let (cgfont, ctfont) = (fontV2.defaultCGFont, fontV2.ctFont)
            XCTAssertNotNil(cgfont)
            XCTAssertNotNil(ctfont)
            let mTable = mathFont.rawMathTable()
            XCTAssertNotNil(mTable)
            self?.testCount += 1
        }
        queue.async(group: group, execute: workitem)
    }
    func testConcurrentThreadsafeMathTableLockScript() throws {
        testCount = 0
        var mathFont: MathFont { .allCases.randomElement()! }
        var size: CGFloat { CGFloat.random(in: 20 ... 40) }
        let mtfonts = Array( 0 ..< 5 ).map { _ in mathFont.mtfont(size: size) }
        for caseNumber in 0 ..< totalCases {
            helperConcurrentMTFontV2MathTableLock(caseNumber, mtfont: mtfonts.randomElement()!, in: executionGroup, on: executionQueue)
        }
        executionGroup.notify(queue: .main) { [weak self] in
            guard let self = self else { return }
            XCTAssertEqual(self.testCount, totalCases)
            print("\(self.testCount) completed =================")
        }
        executionGroup.wait()
    }
    func helperConcurrentMTFontV2MathTableLock(_ count: Int, mtfont: MTFontV2, in group: DispatchGroup, on queue: DispatchQueue) {
        let workitem = DispatchWorkItem {
            let mathTable = mtfont.mathTable as? MTFontMathTableV2
            // each mathTable is initialized once per mtfont with a NSLock.
            // this is even when mathTable is accessed via different threads.
            XCTAssertNotNil(mathTable)
        }
        workitem.notify(queue: .main) { [weak self] in
            // print("\(Thread.isMainThread ? "main" : "global") completed .....")
            self?.testCount += 1
        }
        queue.async(group: group, execute: workitem)
    }
}
