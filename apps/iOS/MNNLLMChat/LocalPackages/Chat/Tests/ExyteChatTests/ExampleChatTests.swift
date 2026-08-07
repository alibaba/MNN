import XCTest
import SwiftUI
@testable import ExyteChat

final class ExampleChatTests: XCTestCase {
    func testSomeInnerLogic() {
        // Given
        let expected: [CGFloat]  = [1, 0, 0, 1]

        // When
        let color = Color(hex: "FF0000")

        // Then

        XCTAssertNotNil(color.cgColor)
        XCTAssertNotNil(color.cgColor!.components)
        for (index, component) in color.cgColor!.components!.enumerated() {
            XCTAssertEqual(component, expected[index], accuracy: 1 / 256)
        }
    }
}
