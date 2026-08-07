//
//  LRUCacheTests.swift
//  LRUCacheTests
//
//  Created by Nick Lockwood on 05/08/2021.
//  Copyright © 2021 Nick Lockwood. All rights reserved.
//

import LRUCache
import XCTest

class LRUCacheTests: XCTestCase {
    func testCountLimit() {
        let cache = LRUCache<Int, Int>(countLimit: 2)
        cache.setValue(0, forKey: 0)
        XCTAssertNotNil(cache.value(forKey: 0))
        cache.setValue(1, forKey: 1)
        cache.setValue(2, forKey: 2)
        XCTAssertNil(cache.value(forKey: 0))
        XCTAssertNotNil(cache.value(forKey: 1))
        XCTAssertNotNil(cache.value(forKey: 2))
        XCTAssertEqual(cache.count, 2)
    }

    func testCostLimit() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 3)
        cache.setValue(0, forKey: 0, cost: 1)
        cache.setValue(1, forKey: 1, cost: 1)
        XCTAssertEqual(cache.count, 2)
        XCTAssertEqual(cache.totalCost, 2)
        cache.setValue(2, forKey: 2, cost: 2)
        XCTAssertNil(cache.value(forKey: 0))
        XCTAssertNotNil(cache.value(forKey: 1))
        XCTAssertNotNil(cache.value(forKey: 2))
        XCTAssertEqual(cache.count, 2)
        XCTAssertEqual(cache.totalCost, 3)
    }

    func testAdjustCountLimit() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 2)
        cache.setValue(0, forKey: 0, cost: 1)
        cache.setValue(1, forKey: 1, cost: 1)
        cache.countLimit = 1
        XCTAssertNil(cache.value(forKey: 0))
        XCTAssertEqual(cache.count, 1)
    }

    func testAdjustCostLimit() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 3)
        cache.setValue(0, forKey: 0, cost: 1)
        cache.setValue(1, forKey: 1, cost: 1)
        cache.setValue(2, forKey: 2, cost: 1)
        cache.totalCostLimit = 2
        XCTAssertNil(cache.value(forKey: 0))
        XCTAssertEqual(cache.count, 2)
        cache.totalCostLimit = 0
        XCTAssert(cache.isEmpty)
    }

    func testRemoveValue() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 2)
        cache.setValue(0, forKey: 0)
        cache.setValue(1, forKey: 1)
        XCTAssertEqual(cache.removeValue(forKey: 0), 0)
        XCTAssertEqual(cache.count, 1)
        XCTAssertNil(cache.removeValue(forKey: 0))
        cache.setValue(nil, forKey: 1)
        XCTAssert(cache.isEmpty)
    }

    func testRemoveAllValues() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 2)
        cache.setValue(0, forKey: 0, cost: 1)
        cache.setValue(1, forKey: 1, cost: 1)
        cache.removeAll()
        XCTAssert(cache.isEmpty)
        XCTAssertEqual(cache.totalCost, 0)
        cache.setValue(0, forKey: 0, cost: 1)
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.totalCost, 1)
    }

    func testOrderedKeys() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 2)
        cache.setValue(0, forKey: 0)
        cache.setValue(1, forKey: 1)
        XCTAssertEqual(cache.orderedKeys, [0, 1])
        cache.setValue(0, forKey: 0)
        XCTAssertEqual(cache.orderedKeys, [1, 0])
        cache.removeAll()
        XCTAssert(cache.orderedKeys.isEmpty)
    }

    func testOrderedValues() {
        let cache = LRUCache<Int, Int>(totalCostLimit: 2)
        cache.setValue(0, forKey: 0)
        cache.setValue(1, forKey: 1)
        XCTAssertEqual(cache.orderedValues, [0, 1])
        cache.setValue(0, forKey: 0)
        XCTAssertEqual(cache.orderedValues, [1, 0])
        cache.removeAll()
        XCTAssert(cache.orderedValues.isEmpty)
    }

    func testReplaceValue() {
        let cache = LRUCache<Int, Int>()
        cache.setValue(0, forKey: 0, cost: 5)
        XCTAssertEqual(cache.value(forKey: 0), 0)
        XCTAssertEqual(cache.totalCost, 5)
        cache.setValue(1, forKey: 0, cost: 3)
        XCTAssertEqual(cache.value(forKey: 0), 1)
        XCTAssertEqual(cache.totalCost, 3)
        cache.setValue(2, forKey: 0, cost: 7)
        XCTAssertEqual(cache.value(forKey: 0), 2)
        XCTAssertEqual(cache.totalCost, 7)
    }

    #if !os(WASI)

    func testConcurrentAccess() {
        let cache = LRUCache<String, Int>()
        let queue = DispatchQueue(label: "stress.test", attributes: .concurrent)
        let group = DispatchGroup()

        let keys = (0 ..< 1000).map { "key\($0)" }
        for _ in 0 ..< 10000 {
            group.enter()
            queue.async {
                let key = keys.randomElement()!
                if Bool.random() {
                    cache.setValue(.random(in: 0 ... 1000), forKey: key)
                } else {
                    _ = cache.value(forKey: key)
                }
                group.leave()
            }
        }

        group.wait()
    }

    @available(*, deprecated, message: "Obsolete")
    func testMemoryWarning() {
        let cache = LRUCache<Int, Int>()
        for i in 0 ..< 100 {
            cache.setValue(i, forKey: i)
        }
        XCTAssertEqual(cache.count, 100)
        NotificationCenter.default.post(
            name: LRUCacheMemoryWarningNotification,
            object: nil
        )
        XCTAssert(cache.isEmpty)
    }

    @available(*, deprecated, message: "Obsolete")
    func testClearsOnMemoryPressureDisabled() {
        let cache = LRUCache<Int, Int>(clearsOnMemoryPressure: false)
        for i in 0 ..< 100 {
            cache.setValue(i, forKey: i)
        }
        XCTAssertEqual(cache.count, 100)
        NotificationCenter.default.post(
            name: LRUCacheMemoryWarningNotification,
            object: nil
        )
        // Cache should NOT be cleared when clearsOnMemoryPressure is false
        XCTAssertEqual(cache.count, 100)
    }

    @available(*, deprecated, message: "Obsolete")
    func testMemoryPressureNotification() {
        // Test with custom notification center to verify event handler execution
        let notificationCenter = NotificationCenter()

        let cache = LRUCache<Int, Int>(
            totalCostLimit: .max,
            countLimit: .max,
            clearsOnMemoryPressure: true
        )

        // Add some values
        for i in 0 ..< 10 {
            cache.setValue(i, forKey: i)
        }
        XCTAssertEqual(cache.count, 10)

        // Post memory warning notification
        notificationCenter.post(
            name: LRUCacheMemoryWarningNotification,
            object: nil
        )

        // Note: This test uses the default notification center, so we can't easily
        // test the event handler execution without modifying the cache implementation
        // The existing testMemoryWarning() already tests the basic functionality

        // Test that the cache responds to memory pressure
        NotificationCenter.default.post(
            name: LRUCacheMemoryWarningNotification,
            object: nil
        )

        // Cache should be empty after memory warning
        XCTAssertTrue(cache.isEmpty)
        XCTAssertEqual(cache.count, 0)
    }

    @available(*, deprecated, message: "Obsolete")
    func testNotificationObserverIsRemoved() {
        #if !os(Linux)
        final class TestNotificationCenter: NotificationCenter, @unchecked Sendable {
            private(set) var observersCount = 0

            override func addObserver(
                forName name: NSNotification.Name?,
                object obj: Any?,
                queue: OperationQueue?,
                using block: @escaping @Sendable (Notification) -> Void
            ) -> NSObjectProtocol {
                defer { observersCount += 1 }
                return super.addObserver(
                    forName: name,
                    object: obj,
                    queue: queue,
                    using: block
                )
            }

            override func removeObserver(_ observer: Any) {
                super.removeObserver(observer)
                observersCount -= 1
            }
        }

        let notificationCenter = TestNotificationCenter()
        var cache: LRUCache<Int, Int>? = .init(notificationCenter: notificationCenter)
        #if compiler(<6.2)
        weak var weakCache = cache
        #else
        weak let weakCache = cache
        #endif
        XCTAssertEqual(1, notificationCenter.observersCount)
        cache = nil
        XCTAssertNil(weakCache)
        XCTAssertEqual(0, notificationCenter.observersCount)
        #endif
    }

    #endif

    func testNoStackOverflowForlargeCache() {
        let cache = LRUCache<Int, Int>()
        for i in 0 ..< 100000 {
            cache.setValue(i, forKey: i)
        }
    }

    func testHasValue() {
        let cache = LRUCache<String, Int>()

        // Test empty cache
        XCTAssertFalse(cache.hasValue(forKey: "test"))

        // Test existing key
        cache.setValue(42, forKey: "test")
        XCTAssertTrue(cache.hasValue(forKey: "test"))

        // Test non-existing key
        XCTAssertFalse(cache.hasValue(forKey: "nonexistent"))

        // Verify hasValue doesn't affect LRU order
        cache.removeAll() // Start fresh
        cache.setValue(1, forKey: "first")
        cache.setValue(2, forKey: "second")
        cache.setValue(3, forKey: "third")

        // Check hasValue doesn't change order
        XCTAssertTrue(cache.hasValue(forKey: "first"))
        XCTAssertEqual(cache.orderedKeys, ["first", "second", "third"])

        // Access value should change order
        _ = cache.value(forKey: "first")
        XCTAssertEqual(cache.orderedKeys, ["second", "third", "first"])
    }

    func testValuesProperty() {
        let cache = LRUCache<String, Int>()

        // Test empty cache
        XCTAssertTrue(cache.values.isEmpty)

        // Add values and test
        cache.setValue(1, forKey: "a")
        cache.setValue(2, forKey: "b")
        cache.setValue(3, forKey: "c")

        let values = Array(cache.values)
        XCTAssertEqual(values.count, 3)
        XCTAssertTrue(values.contains(1))
        XCTAssertTrue(values.contains(2))
        XCTAssertTrue(values.contains(3))

        // Test after removal
        cache.removeValue(forKey: "b")
        let valuesAfterRemoval = Array(cache.values)
        XCTAssertEqual(valuesAfterRemoval.count, 2)
        XCTAssertTrue(valuesAfterRemoval.contains(1))
        XCTAssertTrue(valuesAfterRemoval.contains(3))
        XCTAssertFalse(valuesAfterRemoval.contains(2))
    }

    func testPropertyGetters() {
        let cache = LRUCache<Int, String>(totalCostLimit: 100, countLimit: 50)

        // Test initial values
        XCTAssertEqual(cache.totalCostLimit, 100)
        XCTAssertEqual(cache.countLimit, 50)

        // Test after setting values
        cache.totalCostLimit = 200
        cache.countLimit = 75

        XCTAssertEqual(cache.totalCostLimit, 200)
        XCTAssertEqual(cache.countLimit, 75)
    }

    func testNilValueHandling() {
        let cache = LRUCache<String, Int>()

        // Test setting nil for non-existent key
        cache.setValue(nil, forKey: "nonexistent")
        XCTAssertFalse(cache.hasValue(forKey: "nonexistent"))
        XCTAssertEqual(cache.count, 0)

        // Test setting nil for existing key
        cache.setValue(42, forKey: "test")
        XCTAssertTrue(cache.hasValue(forKey: "test"))
        XCTAssertEqual(cache.count, 1)

        cache.setValue(nil, forKey: "test")
        XCTAssertFalse(cache.hasValue(forKey: "test"))
        XCTAssertEqual(cache.count, 0)
    }

    func testNegativeCostHandling() {
        let cache = LRUCache<String, Int>()

        // Test setting negative cost
        cache.setValue(42, forKey: "test", cost: -5)
        XCTAssertEqual(cache.value(forKey: "test"), 42)
        XCTAssertEqual(cache.totalCost, -5)

        // Test replacing with positive cost
        cache.setValue(43, forKey: "test", cost: 10)
        XCTAssertEqual(cache.value(forKey: "test"), 43)
        XCTAssertEqual(cache.totalCost, 10)
    }

    func testEmptyCacheOperations() {
        let cache = LRUCache<String, Int>()

        // Test all operations on empty cache
        XCTAssertNil(cache.value(forKey: "test"))
        XCTAssertNil(cache.removeValue(forKey: "test"))
        XCTAssertFalse(cache.hasValue(forKey: "test"))
        XCTAssertTrue(cache.isEmpty)
        XCTAssertEqual(cache.count, 0)
        XCTAssertEqual(cache.totalCost, 0)
        XCTAssertTrue(cache.keys.isEmpty)
        XCTAssertTrue(cache.values.isEmpty)
        XCTAssertTrue(cache.orderedKeys.isEmpty)
        XCTAssertTrue(cache.orderedValues.isEmpty)

        // Test removeAll on empty cache
        cache.removeAll()
        XCTAssertTrue(cache.isEmpty)
    }

    func testStringKeyAndValueTypes() {
        let cache = LRUCache<String, String>()

        cache.setValue("hello", forKey: "greeting")
        cache.setValue("world", forKey: "object")

        XCTAssertEqual(cache.value(forKey: "greeting"), "hello")
        XCTAssertEqual(cache.value(forKey: "object"), "world")
        XCTAssertEqual(cache.count, 2)

        let keys = Array(cache.keys)
        XCTAssertTrue(keys.contains("greeting"))
        XCTAssertTrue(keys.contains("object"))

        let values = Array(cache.values)
        XCTAssertTrue(values.contains("hello"))
        XCTAssertTrue(values.contains("world"))
    }

    func testObjectValueTypes() {
        struct TestObject: Equatable {
            let id: Int
            let name: String
        }

        let cache = LRUCache<Int, TestObject>()

        let obj1 = TestObject(id: 1, name: "first")
        let obj2 = TestObject(id: 2, name: "second")

        cache.setValue(obj1, forKey: 1)
        cache.setValue(obj2, forKey: 2)

        XCTAssertEqual(cache.value(forKey: 1), obj1)
        XCTAssertEqual(cache.value(forKey: 2), obj2)
        XCTAssertEqual(cache.count, 2)
    }

    func testCostLimitWithZeroCost() {
        let cache = LRUCache<String, Int>(totalCostLimit: 10)

        // Test adding items with zero cost
        cache.setValue(1, forKey: "a", cost: 0)
        cache.setValue(2, forKey: "b", cost: 0)
        cache.setValue(3, forKey: "c", cost: 0)

        // All should be allowed since cost is 0
        XCTAssertEqual(cache.count, 3)
        XCTAssertEqual(cache.totalCost, 0)

        // Test adding item with positive cost within limit
        cache.setValue(4, forKey: "d", cost: 5)
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(cache.totalCost, 5)

        // Test adding item that would exceed limit
        cache.setValue(5, forKey: "e", cost: 10)
        // Should trigger eviction of oldest items
        XCTAssertLessThanOrEqual(cache.totalCost, 10)
    }

    func testCacheWithZeroLimits() {
        let cache = LRUCache<String, Int>(totalCostLimit: 0, countLimit: 0)

        // Test that items are immediately evicted when limits are 0
        cache.setValue(1, forKey: "a", cost: 0)
        XCTAssertEqual(cache.count, 0)
        XCTAssertNil(cache.value(forKey: "a"))

        // Even with positive cost, should be evicted
        cache.setValue(2, forKey: "b", cost: 5)
        XCTAssertEqual(cache.count, 0)
        XCTAssertNil(cache.value(forKey: "b"))
    }

    func testLRUOrderConsistency() {
        let cache = LRUCache<String, Int>(countLimit: 3)

        // Add items
        cache.setValue(1, forKey: "a")
        cache.setValue(2, forKey: "b")
        cache.setValue(3, forKey: "c")

        XCTAssertEqual(cache.orderedKeys, ["a", "b", "c"])

        // Access middle item - should move to end
        _ = cache.value(forKey: "b")
        XCTAssertEqual(cache.orderedKeys, ["a", "c", "b"])

        // Access first item - should move to end
        _ = cache.value(forKey: "a")
        XCTAssertEqual(cache.orderedKeys, ["c", "b", "a"])

        // Add new item - should evict first (c)
        cache.setValue(4, forKey: "d")
        XCTAssertEqual(cache.orderedKeys, ["b", "a", "d"])
        XCTAssertNil(cache.value(forKey: "c"))
        XCTAssertNotNil(cache.value(forKey: "b"))
        XCTAssertNotNil(cache.value(forKey: "a"))
        XCTAssertNotNil(cache.value(forKey: "d"))
    }
}
