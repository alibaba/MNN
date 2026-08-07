#if !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class ViewControllerTests: XCTestCase {
    func testViewController() {
        XCTAssertViewIntrospection(of: PlatformViewController.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            TabView {
                NavigationView {
                    Text("Root").frame(maxWidth: .infinity, maxHeight: .infinity).background(Color.red)
                        .introspect(
                            .viewController,
                            on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                            customize: spy2
                        )
                }
                .navigationViewStyle(.stack)
                .tabItem {
                    Image(systemName: "1.circle")
                    Text("Tab 1")
                }
                .introspect(
                    .viewController,
                    on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                    customize: spy1
                )
            }
            .introspect(
                .viewController,
                on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2),
                customize: spy0
            )
        } extraAssertions: {
            #if !os(visionOS)
            XCTAssert($0[safe: 0] is UITabBarController)
            #endif
            XCTAssert($0[safe: 1] is UINavigationController)
            XCTAssert(String(describing: $0[safe: 2]).contains("UIHostingController"))
            XCTAssert($0[safe: 1] === $0[safe: 2]?.parent)
        }
    }
}
#endif
