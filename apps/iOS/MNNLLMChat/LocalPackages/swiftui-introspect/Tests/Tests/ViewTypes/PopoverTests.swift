#if !os(tvOS) && !os(macOS) && !targetEnvironment(macCatalyst)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class PopoverTests: XCTestCase {
    func testPopover() throws {
        XCTAssertViewIntrospection(of: UIPopoverPresentationController.self) { spies in
            let spy0 = spies[0]

            Text("Root")
                .popover(isPresented: .constant(true)) {
                    Text("Popover")
                        .introspect(
                            .popover,
                            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                            customize: spy0
                        )
                }
        }
    }
}
#endif
