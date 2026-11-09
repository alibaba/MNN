#if !os(macOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class NavigationViewWithStackStyleTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformNavigationViewWithStackStyle = UINavigationController
    #endif

    func testNavigationViewWithStackStyle() {
        XCTAssertViewIntrospection(of: PlatformNavigationViewWithStackStyle.self) { spies in
            let spy = spies[0]

            NavigationView {
                ZStack {
                    Color.red
                    Text("Something")
                }
            }
            .navigationViewStyle(.stack)
            #if os(iOS) || os(tvOS) || os(visionOS)
            .introspect(.navigationView(style: .stack), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy)
            #endif
        }
    }

    func testNavigationViewWithStackStyleAsAncestor() {
        XCTAssertViewIntrospection(of: PlatformNavigationViewWithStackStyle.self) { spies in
            let spy = spies[0]

            NavigationView {
                ZStack {
                    Color.red
                    Text("Something")
                        #if os(iOS) || os(tvOS) || os(visionOS)
                        .introspect(.navigationView(style: .stack), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor, customize: spy)
                        #endif
                }
            }
            .navigationViewStyle(.stack)
        }
    }
}
#endif
