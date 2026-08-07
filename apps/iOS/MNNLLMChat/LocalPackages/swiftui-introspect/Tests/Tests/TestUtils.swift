import SwiftUI
import XCTest

#if canImport(UIKit)
@MainActor
enum TestUtils {
    #if targetEnvironment(macCatalyst) || os(visionOS)
    static let window = UIWindow(frame: CGRect(x: 0, y: 0, width: 480, height: 300))
    #else
    static let window = UIWindow(frame: UIScreen.main.bounds)
    #endif

    static func present(view: some View, file: StaticString = #file, line: UInt = #line) {
        if let window =
            UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).first?.windows.first
            ??
            UIApplication.shared.windows.first
        {
            window.rootViewController = UIHostingController(rootView: view)
        } else {
            window.rootViewController = UIHostingController(rootView: view)
            window.makeKeyAndVisible()
            window.layoutIfNeeded()
        }
    }
}
#elseif canImport(AppKit)
@MainActor
enum TestUtils {
    private static let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 480, height: 300),
        styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
        backing: .buffered,
        defer: true
    )

    static func present(view: some View) {
        window.contentView = NSHostingView(rootView: view)
        window.makeKeyAndOrderFront(nil)
        window.layoutIfNeeded()
    }
}
#endif

@MainActor
func XCTAssertViewIntrospection<Entity: AnyObject>(
    of type: Entity.Type,
    @ViewBuilder view: (Spies<Entity>) -> some View,
    extraAssertions: ([Entity]) -> Void = { _ in },
    file: StaticString = #file,
    line: UInt = #line
) {
    let spies = Spies<Entity>()
    let view = view(spies)
    TestUtils.present(view: view)
    XCTWaiter(delegate: spies).wait(for: spies.expectations.values.map(\.0), timeout: 3)
    extraAssertions(spies.entities.sorted(by: { $0.key < $1.key }).map(\.value))
}

final class Spies<Entity: AnyObject>: NSObject, XCTWaiterDelegate {
    private(set) var entities: [Int: Entity] = [:]
    private(set) var expectations: [ObjectIdentifier: (XCTestExpectation, StaticString, UInt)] = [:]

    subscript(
        number: Int,
        file: StaticString = #file,
        line: UInt = #line
    ) -> (Entity) -> Void {
        let expectation = XCTestExpectation()
        expectations[ObjectIdentifier(expectation)] = (expectation, file, line)
        return { [self] in
            if let entity = entities[number] {
                XCTAssert(entity === $0, "Found view was overriden by another view", file: file, line: line)
            }
            entities[number] = $0
            expectation.fulfill()
        }
    }

    func waiter(
        _ waiter: XCTWaiter,
        didTimeoutWithUnfulfilledExpectations unfulfilledExpectations: [XCTestExpectation]
    ) {
        for expectation in unfulfilledExpectations {
            let (_, file, line) = expectations[ObjectIdentifier(expectation)]!
            XCTFail("Spy not called", file: file, line: line)
        }
    }

    func nestedWaiter(
        _ waiter: XCTWaiter,
        wasInterruptedByTimedOutWaiter outerWaiter: XCTWaiter
    ) {
        XCTFail("wasInterruptedByTimedOutWaiter")
    }

    func waiter(
        _ waiter: XCTWaiter,
        fulfillmentDidViolateOrderingConstraintsFor expectation: XCTestExpectation,
        requiredExpectation: XCTestExpectation
    ) {
        XCTFail("fulfillmentDidViolateOrderingConstraintsFor")
    }

    func waiter(
        _ waiter: XCTWaiter,
        didFulfillInvertedExpectation expectation: XCTestExpectation
    ) {
        XCTFail("didFulfillInvertedExpectation")
    }
}

extension Collection {
    subscript(safe index: Index, file: StaticString = #file, line: UInt = #line) -> Element? {
        guard indices.contains(index) else {
            XCTFail("Index \(index) is out of bounds", file: file, line: line)
            return nil
        }
        return self[index]
    }
}
