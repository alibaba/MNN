#if !os(tvOS)
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 16, macOS 12, *)
@MainActor
final class TableTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformTable = UICollectionView
    #elseif canImport(AppKit)
    typealias PlatformTable = NSTableView
    #endif

    func testTable() throws {
        guard #available(iOS 16, macOS 12, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformTable.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TipTable()
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy0)
                    #endif

                TipTable()
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy1)
                    #endif

                TipTable()
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy2)
                    #endif
            }
        }
    }

    func testTableWithInsetStyle() throws {
        guard #available(iOS 16, macOS 12, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformTable.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TipTable()
                    .tableStyle(.inset)
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy0)
                    #endif

                TipTable()
                    .tableStyle(.inset)
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy1)
                    #endif

                TipTable()
                    .tableStyle(.inset)
                    #if os(iOS) || os(visionOS)
                    .introspect(.table, on: .iOS(.v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy2)
                    #endif
            }
        }
    }

    #if os(macOS)
    func testTableWithBorderedStyle() throws {
        guard #available(macOS 12, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformTable.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                TipTable()
                    .tableStyle(.bordered)
                    #if os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy0)
                    #endif

                TipTable()
                    .tableStyle(.bordered)
                    #if os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy1)
                    #endif

                TipTable()
                    .tableStyle(.bordered)
                    #if os(macOS)
                    .introspect(.table, on: .macOS(.v12, .v13, .v14), customize: spy2)
                    #endif
            }
        }
    }
    #endif
}

@available(iOS 16, macOS 12, *)
extension TableTests {
    struct TipTable: View {
        struct Purchase: Identifiable {
            let price: Decimal
            let id = UUID()
        }

        var body: some View {
            Table(of: Purchase.self) {
                TableColumn("Base price") { purchase in
                    Text(purchase.price, format: .currency(code: "USD"))
                }
                TableColumn("With 15% tip") { purchase in
                    Text(purchase.price * 1.15, format: .currency(code: "USD"))
                }
                TableColumn("With 20% tip") { purchase in
                    Text(purchase.price * 1.2, format: .currency(code: "USD"))
                }
                TableColumn("With 25% tip") { purchase in
                    Text(purchase.price * 1.25, format: .currency(code: "USD"))
                }
            } rows: {
                TableRow(Purchase(price: 20))
                TableRow(Purchase(price: 50))
                TableRow(Purchase(price: 75))
            }
        }
    }
}
#endif
