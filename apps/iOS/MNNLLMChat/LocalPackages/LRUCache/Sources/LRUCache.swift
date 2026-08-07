//
//  LRUCache.swift
//  LRUCache
//
//  Created by Nick Lockwood on 05/08/2021.
//  Copyright © 2021 Nick Lockwood. All rights reserved.
//
//  Distributed under the permissive MIT license
//  Get the latest version from here:
//
//  https://github.com/nicklockwood/LRUCache
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.
//

import Foundation

#if !os(WASI)

@available(*, deprecated, message: "Do not use")
public let LRUCacheMemoryWarningNotification: NSNotification.Name = _notification

#endif

public final class LRUCache<Key: Hashable & Sendable, Value>: @unchecked Sendable {
    private var _values: [Key: Container] = [:]
    private var _countLimit: Int
    private var _totalCost: Int = 0
    private var _totalCostLimit: Int
    private unowned(unsafe) var head: Container?
    private unowned(unsafe) var tail: Container?
    private let clearsOnMemoryPressure: Bool
    private let lock: NSLock = .init()

    #if os(iOS) || os(macOS) || os(tvOS) || os(watchOS) || os(visionOS)
    private let memoryPressureSource: DispatchSourceMemoryPressure?
    #endif

    /// Initialize the cache with the specified `totalCostLimit` and `countLimit`
    public init(
        totalCostLimit: Int = .max,
        countLimit: Int = .max,
        clearsOnMemoryPressure: Bool = true
    ) {
        self._totalCostLimit = totalCostLimit
        self._countLimit = countLimit
        self.clearsOnMemoryPressure = clearsOnMemoryPressure

        #if os(iOS) || os(macOS) || os(tvOS) || os(watchOS) || os(visionOS)
        if clearsOnMemoryPressure {
            self.memoryPressureSource = DispatchSource
                .makeMemoryPressureSource(eventMask: [.warning, .critical], queue: .global())
            memoryPressureSource?.setEventHandler { [weak self] in
                self?.removeAll()
            }
            memoryPressureSource?.resume()
        } else {
            self.memoryPressureSource = nil
        }
        #endif

        #if !os(WASI)
        if clearsOnMemoryPressure {
            registerNotifications(for: .default)
        }
        #endif
    }

    #if !os(WASI)

    @available(*, deprecated, message: "Do not use")
    public convenience init(
        totalCostLimit: Int = .max,
        countLimit: Int = .max,
        notificationCenter: NotificationCenter
    ) {
        self.init(totalCostLimit: totalCostLimit, countLimit: countLimit)
        registerNotifications(for: notificationCenter)
    }

    private var token: AnyObject?
    private var notificationCenter: NotificationCenter = .default

    func registerNotifications(for notificationCenter: NotificationCenter) {
        token.map(self.notificationCenter.removeObserver)
        self.notificationCenter = notificationCenter
        token = notificationCenter.addObserver(
            forName: _notification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            self?.removeAll()
        }
    }

    deinit {
        token.map(notificationCenter.removeObserver)

        #if os(iOS) || os(macOS) || os(tvOS) || os(watchOS) || os(visionOS)
        self.memoryPressureSource?.cancel()
        #endif
    }

    #endif
}

public extension LRUCache {
    /// The current total cost of values in the cache
    var totalCost: Int {
        atomic { _totalCost }
    }

    /// The maximum total cost permitted
    var totalCostLimit: Int {
        get { atomic { _totalCostLimit } }
        set {
            atomic {
                _totalCostLimit = newValue
                clean()
            }
        }
    }

    /// The number of values currently stored in the cache
    var count: Int {
        atomic { _values.count }
    }

    /// The maximum number of values permitted
    var countLimit: Int {
        get { atomic { _countLimit } }
        set {
            atomic {
                _countLimit = newValue
                clean()
            }
        }
    }

    /// Is the cache empty?
    var isEmpty: Bool {
        atomic { _values.isEmpty }
    }

    /// All keys in the cache, in no particular order
    var keys: some Collection<Key> {
        atomic { _values.keys }
    }

    /// All values in the cache, in no particular order
    var values: some Collection<Value> {
        atomic { _values.values.map(\.value) }
    }

    /// All keys in the cache, ordered from least recently used to most recently used
    /// Note: this is orders of magnitude slower to compute than `keys`
    var orderedKeys: [Key] {
        atomic {
            var keys = [Key]()
            keys.reserveCapacity(_values.count)
            var next = head
            while let container = next {
                keys.append(container.key)
                next = container.next
            }
            return keys
        }
    }

    /// All values in the cache, ordered from least recently used to most recently used
    /// Note: this is orders of magnitude slower to compute than `values`
    var orderedValues: [Value] {
        atomic {
            var values = [Value]()
            values.reserveCapacity(_values.count)
            var next = head
            while let container = next {
                values.append(container.value)
                next = container.next
            }
            return values
        }
    }

    /// All keys in the cache, ordered from least recently used to most recently used
    @available(*, deprecated, renamed: "orderedKeys")
    var allKeys: [Key] { orderedKeys }

    /// All values in the cache, ordered from least recently used to most recently used
    @available(*, deprecated, renamed: "orderedValues")
    var allValues: [Value] { orderedValues }

    /// Insert a value into the cache with optional `cost` and mark it as most recently used
    func setValue(_ value: Value?, forKey key: Key, cost: Int = 0) {
        guard let value else {
            removeValue(forKey: key)
            return
        }
        atomic {
            if let container = _values[key] {
                container.value = value
                _totalCost += cost - container.cost
                container.cost = cost
                remove(container)
                append(container)
            } else {
                let container = Container(
                    value: value,
                    cost: cost,
                    key: key
                )
                _totalCost += cost
                _values[key] = container
                append(container)
            }
            clean()
        }
    }

    /// Check if a value exists in the cache without affecting how recently it was used
    func hasValue(forKey key: Key) -> Bool {
        atomic { _values[key] != nil }
    }

    /// Fetch a value from the cache and mark it as most recently used
    func value(forKey key: Key) -> Value? {
        atomic {
            if let container = _values[key] {
                remove(container)
                append(container)
                return container.value
            }
            return nil
        }
    }

    /// Remove a value  from the cache and return it
    @discardableResult func removeValue(forKey key: Key) -> Value? {
        atomic {
            guard let container = _values.removeValue(forKey: key) else {
                return nil
            }
            remove(container)
            _totalCost -= container.cost
            return container.value
        }
    }

    /// Remove all values from the cache
    func removeAll() {
        atomic {
            _values.removeAll()
            head = nil
            tail = nil
            _totalCost = 0
        }
    }

    /// Remove all values from the cache
    @available(*, deprecated, renamed: "removeAll")
    func removeAllValues() { removeAll() }
}

#if canImport(UIKit) && !os(watchOS)
import UIKit

private let _notification: NSNotification.Name = UIApplication.didReceiveMemoryWarningNotification
#elseif !os(WASI)
private let _notification: NSNotification.Name = .init("LRUCacheMemoryWarningNotification")
#endif

private extension LRUCache {
    final class Container {
        var value: Value
        var cost: Int
        let key: Key
        unowned(unsafe) var prev: Container?
        unowned(unsafe) var next: Container?

        init(value: Value, cost: Int, key: Key) {
            self.value = value
            self.cost = cost
            self.key = key
        }
    }

    /// Atomic access
    func atomic<T>(_ action: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return action()
    }

    /// Remove container from list (must be called inside lock)
    func remove(_ container: Container) {
        if head === container {
            head = container.next
        }
        if tail === container {
            tail = container.prev
        }
        container.next?.prev = container.prev
        container.prev?.next = container.next
        container.next = nil
    }

    /// Append container to list (must be called inside lock)
    func append(_ container: Container) {
        assert(container.next == nil)
        if head == nil {
            head = container
        }
        container.prev = tail
        tail?.next = container
        tail = container
    }

    /// Remove expired values (must be called inside lock)
    func clean() {
        while _totalCost > _totalCostLimit || _values.count > _countLimit,
              let container = head
        {
            remove(container)
            _values.removeValue(forKey: container.key)
            _totalCost -= container.cost
        }
    }
}
