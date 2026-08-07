import SwiftUI
import SwiftUIIntrospect
import XCTest

@MainActor
final class NavigationViewWithColumnsStyleTests: XCTestCase {
    #if canImport(UIKit) && (os(iOS) || os(visionOS))
    typealias PlatformNavigationViewWithColumnsStyle = UISplitViewController
    #elseif canImport(UIKit) && os(tvOS)
    typealias PlatformNavigationViewWithColumnsStyle = UINavigationController
    #elseif canImport(AppKit)
    typealias PlatformNavigationViewWithColumnsStyle = NSSplitView
    #endif

    func testNavigationViewWithColumnsStyle() {
        XCTAssertViewIntrospection(of: PlatformNavigationViewWithColumnsStyle.self) { spies in
            let spy = spies[0]

            NavigationView {
                ZStack {
                    Color.red
                    Text("Something")
                }
            }
            .navigationViewStyle(DoubleColumnNavigationViewStyle())
            #if os(iOS) || os(visionOS)
            .introspect(.navigationView(style: .columns), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy)
            #elseif os(tvOS)
            .introspect(.navigationView(style: .columns), on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), customize: spy)
            #elseif os(macOS)
            .introspect(.navigationView(style: .columns), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), customize: spy)
            #endif
        }
    }

    func testNavigationViewWithColumnsStyleAsAncestor() {
        XCTAssertViewIntrospection(of: PlatformNavigationViewWithColumnsStyle.self) { spies in
            let spy = spies[0]

            NavigationView {
                ZStack {
                    Color.red
                    Text("Something")
                        #if os(iOS) || os(visionOS)
                        .introspect(.navigationView(style: .columns), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), scope: .ancestor, customize: spy)
                        #elseif os(tvOS)
                        .introspect(.navigationView(style: .columns), on: .tvOS(.v13, .v14, .v15, .v16, .v17, .v18), scope: .ancestor, customize: spy)
                        #elseif os(macOS)
                        .introspect(.navigationView(style: .columns), on: .macOS(.v10_15, .v11, .v12, .v13, .v14), scope: .ancestor, customize: spy)
                        #endif
                }
            }
            .navigationViewStyle(DoubleColumnNavigationViewStyle())
            #if os(iOS)
            // NB: this is necessary for ancestor introspection to work, because initially on iPad the "Customized" text isn't shown as it's hidden in the sidebar. This is why ancestor introspection is discouraged for most situations and it's opt-in.
            .introspect(.navigationView(style: .columns), on: .iOS(.v13, .v14, .v15, .v16, .v17, .v18)) {
                $0.preferredDisplayMode = .oneOverSecondary
            }
            #endif
        }
    }
}
