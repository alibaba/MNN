//
//  LRUPerformanceTests.swift
//  LRUCacheTests
//
//  Created by Nick Lockwood on 05/08/2021.
//  Copyright © 2021 Nick Lockwood. All rights reserved.
//

import LRUCache
import XCTest

class LRUPerformanceTests: XCTestCase {
    let iterations = 10000

    private func populateCache(_ cache: LRUCache<Int, Int>) {
        for i in 0 ..< iterations {
            cache.setValue(.random(in: .min ... .max), forKey: i)
        }
    }

    private func createCache(populated: Bool) -> LRUCache<Int, Int> {
        let cache = LRUCache<Int, Int>()
        if populated {
            populateCache(cache)
        }
        return cache
    }

    private func createCaches(_ count: Int = 10, populated: Bool) -> [LRUCache<Int, Int>] {
        var caches = [LRUCache<Int, Int>]()
        for _ in 0 ... count {
            caches.append(createCache(populated: populated))
        }
        return caches
    }

    func testInsertionPerformance() {
        var caches = createCaches(populated: false)
        var result: Any?
        measure {
            let cache = caches.popLast()!
            populateCache(cache)
            result = cache
        }
        XCTAssertNotNil(result)
    }

    func testReinsertionPerformance() {
        let cache = createCache(populated: true)
        measure {
            populateCache(cache)
        }
        XCTAssertEqual(cache.count, iterations)
    }

    func testLookupPerformance() {
        let cache = createCache(populated: true)
        var values = [Int?](repeating: nil, count: iterations)
        measure {
            for i in 0 ..< iterations {
                values[i] = cache.value(forKey: i)
            }
        }
        XCTAssert(values.allSatisfy { $0 != nil })
    }

    func testRemovalPerformance() {
        var caches = createCaches(populated: true)
        var values = [Int?](repeating: nil, count: iterations)
        measure {
            let cache = caches.removeLast()
            for i in 0 ..< iterations {
                values[i] = cache.removeValue(forKey: i)
            }
        }
        XCTAssert(values.allSatisfy { $0 != nil })
    }

    func testOverflowInsertionPerformance() {
        let cache = createCache(populated: false)
        cache.countLimit = 1000
        measure {
            populateCache(cache)
        }
        XCTAssertEqual(cache.count, 1000)
    }

    func testKeysPerformance() {
        let cache = createCache(populated: true)
        var keys: (any Collection<Int>)?
        measure {
            for _ in 0 ..< iterations {
                keys = cache.keys
            }
        }
        XCTAssertEqual(keys?.count, iterations)
    }

    func testValuesPerformance() {
        let cache = createCache(populated: true)
        var values: (any Collection<Int>)?
        measure {
            for _ in 0 ..< iterations {
                values = cache.keys
            }
        }
        XCTAssertEqual(values?.count, iterations)
    }

    #if !os(WASI)

    func testConcurrentAccess() {
        let cache = LRUCache<String, Int>()
        measure {
            let queue = DispatchQueue(label: "stress.test", attributes: .concurrent)
            let group = DispatchGroup()

            let keys = (0 ..< 1000).map { "key\($0)" }
            for _ in 0 ..< iterations {
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
    }

    #endif

    #if os(macOS) || os(iOS)

    @available(iOS 13.0, *)
    func testOrderedKeysPerformance() {
        let options = XCTMeasureOptions()
        options.iterationCount = 1
        let cache = createCache(populated: true)
        var keys: (any Collection<Int>)?
        measure(options: options) {
            for _ in 0 ..< iterations {
                keys = cache.orderedKeys
            }
        }
        XCTAssertEqual(keys?.count, iterations)
    }

    @available(iOS 13.0, *)
    func testOrderedValuesPerformance() {
        let options = XCTMeasureOptions()
        options.iterationCount = 1
        let cache = createCache(populated: true)
        var values: (any Collection<Int>)?
        measure(options: options) {
            for _ in 0 ..< iterations {
                values = cache.orderedValues
            }
        }
        XCTAssertEqual(values?.count, iterations)
    }

    #endif
}
