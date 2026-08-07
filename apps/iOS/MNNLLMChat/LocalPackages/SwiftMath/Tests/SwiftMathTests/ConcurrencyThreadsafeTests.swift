//
//  ConcurrencyThreadsafeTests.swift
//  
//
//  Created by Peter Tang on 26/9/2023.
//

import XCTest
@testable import SwiftMath

final class ConcurrencyThreadsafeTests: XCTestCase {
    
    private let executionQueue = DispatchQueue(label: "com.swiftmath.concurrencytests", attributes: .concurrent)
    private let executionGroup = DispatchGroup()
    
    let totalCases = 20
    var testCount = 0
    
    func testSwiftMathConcurrentScript() throws {
        for caseNumber in 0 ..< totalCases {
            helperConcurrency(caseNumber, in: executionGroup, on: executionQueue) {
                let result1 = getInterElementSpaces()
                let result2 = MTMathAtomFactory.delimValueToName
                let result3 = MTMathAtomFactory.accentValueToName
                let result4 = MTMathAtomFactory.textToLatexSymbolName
                XCTAssertNotNil(result1)
                XCTAssertNotNil(result2)
                XCTAssertNotNil(result3)
                XCTAssertNotNil(result4)
            }
        }
//        executionGroup.notify(queue: .main) { [weak self] in
//            // print("All test cases completed: \(self?.testCount ?? 0)")
//        }
        executionGroup.wait()
    }
    func helperConcurrency(_ count: Int, in group: DispatchGroup, on queue: DispatchQueue, _ testClosure: @escaping () -> (Void)) {
        let workitem = DispatchWorkItem {
            testClosure()
        }
        workitem.notify(queue: .main) { [weak self] in
            self?.testCount += 1
        }
        queue.async(group: group, execute: workitem)
    }

}
