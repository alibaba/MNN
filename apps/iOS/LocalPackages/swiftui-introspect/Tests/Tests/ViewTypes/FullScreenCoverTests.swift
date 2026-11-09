#if !os(macOS) && !targetEnvironment(macCatalyst)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, tvOS 14, *)
@MainActor
final class FullScreenCoverTests: XCTestCase {
    func testPresentationAsFullScreenCover() throws {
        guard #available(iOS 14, tvOS 14, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: UIPresentationController.self) { spies in
            let spy0 = spies[0]

            Text("Root")
                .fullScreenCover(isPresented: .constant(true)) {
                    Text("Content")
                        #if os(iOS) || os(tvOS) || os(visionOS)
                        .introspect(
                            .fullScreenCover,
                            on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                            customize: spy0
                        )
                        #endif
                }
        }
    }
}
#endif
