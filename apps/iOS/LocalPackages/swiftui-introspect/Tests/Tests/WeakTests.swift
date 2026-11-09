@_spi(Advanced) import SwiftUIIntrospect
import XCTest

final class WeakTests: XCTestCase {
    final class Foo {}

    var strongFoo: Foo? = Foo()

    func testInit_nil() {
        @Weak var weakFoo: Foo?
        XCTAssertNil(weakFoo)
    }

    func testInit_nonNil() {
        @Weak var weakFoo: Foo? = strongFoo
        XCTAssertIdentical(weakFoo, strongFoo)
    }

    func testAssignment_nilToNil() {
        @Weak var weakFoo: Foo?
        weakFoo = nil
        XCTAssertNil(weakFoo)
    }

    func testAssignment_nilToNonNil() {
        @Weak var weakFoo: Foo?
        let otherFoo = Foo()
        weakFoo = otherFoo
        XCTAssertIdentical(weakFoo, otherFoo)
    }

    func testAssignment_nonNilToNil() {
        @Weak var weakFoo: Foo? = strongFoo
        weakFoo = nil
        XCTAssertNil(weakFoo)
    }

    func testAssignment_nonNilToNonNil() {
        @Weak var weakFoo: Foo? = strongFoo
        let otherFoo = Foo()
        weakFoo = otherFoo
        XCTAssertIdentical(weakFoo, otherFoo)
    }

    func testIndirectAssignment_nonNilToNil() {
        @Weak var weakFoo: Foo? = strongFoo
        strongFoo = nil
        XCTAssertNil(weakFoo)
    }

    func testIndirectAssignment_nonNilToNonNil() {
        @Weak var weakFoo: Foo? = strongFoo
        strongFoo = Foo()
        XCTAssertNil(weakFoo)
    }
}
