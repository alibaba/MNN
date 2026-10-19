[![Build](https://github.com/nicklockwood/LRUCache/actions/workflows/build.yml/badge.svg)](https://github.com/nicklockwood/LRUCache/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/nicklockwood/LRUCache/graphs/badge.svg)](https://codecov.io/gh/nicklockwood/LRUCache)
[![Platforms](https://img.shields.io/badge/platforms-iOS%20|%20Mac%20|%20tvOS%20|%20watchOS%20|%20Linux-lightgray.svg)]()
[![Swift 5.7](https://img.shields.io/badge/swift-5.7-red.svg?style=flat)](https://developer.apple.com/swift)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)
[![Mastodon](https://img.shields.io/badge/mastodon-@nicklockwood@mastodon.social-636dff.svg)](https://mastodon.social/@nicklockwood)

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Concurrency](#concurrency)
- [Performance](#performance)
- [Credits](#credits)


# Introduction

**LRUCache** is an open-source replacement for [`NSCache`](https://developer.apple.com/library/mac/documentation/cocoa/reference/NSCache_Class/Reference/Reference.html) that behaves in a predictable, debuggable way. **LRUCache** is an LRU (Least-Recently-Used) cache, meaning that objects will be discarded oldest-first based on the last time they were accessed. **LRUCache** will automatically empty itself in the event of a memory warning.


# Installation

LRUCache is packaged as a framework that you can import into your Xcode project. You can install this manually, or by using Swift Package Manager.

**Note:** LRUCache requires Xcode 14+ to build, and runs on iOS/tvOS 13+, watchOS 6+ or macOS 10.15+.

To install using Swift Package Manage, add this to the `dependencies:` section in your Package.swift file:

```swift
.package(url: "https://github.com/nicklockwood/LRUCache.git", .upToNextMinor(from: "1.2.1")),
```


# Usage

You can create an instance of **LRUCache** as follows:

```swift
let cache = LRUCache<String, Int>()
```

This would create a cache of unlimited size, containing `Int` values keyed by `String`. To add a value to the cache, use:

```swift
cache.setValue(99, forKey: "foo")
```

To fetch a cached value, use:

```swift
let value = cache.value(forKey: "foo") // Returns nil if value not found
```

To fetch *all* the keys or values stored in the cache, use:

```swift
let keys = cache.keys
let values = cache.values
```

You can limit the cache size by count and/or *cost*. This can be done at initialization time:

```swift
let cache = LRUCache<URL, Date>(totalCostLimit: 1000, countLimit: 100)
```

Or after the cache has been created:

```swift
cache.countLimit = 100 // Limit the cache to 100 elements
cache.totalCostLimit = 1000 // Limit the cache to 1000 total cost
```

The cost is an arbitrary measure of storage cost for a cached value, defined by your application. For a file or data cache you might base cost on the size in bytes, or any metric you like. To specify the cost of a stored value, use the optional `cost` parameter:

```swift
cache.setValue(data, forKey: "foo", cost: data.count)
```

Values will be removed from the cache automatically when either the count or cost limits are exceeded. You can also remove values explicitly by using:

```swift
let value = cache.removeValue(forKey: "foo")
```

Or by setting it to `nil`:

```swift
cache.setValue(nil, forKey: "foo")
```

You can remove all values from the cache immediately with:

```swift
cache.removeAll()
```

On Apple platforms (macOS, iOS, iPadOS, visionOS, tvOS, watchOS) the cache will be emptied automatically in the event of a memory warning.


# Concurrency

**LRUCache** is `Sendable` and uses `NSLock` internally to ensure mutations are atomic. It is therefore safe to access a single cache instance from multiple threads.


# Performance

Reading, writing and removing entries from the cache are performed in constant time. When the cache is full, insertion time degrades slightly due to the need to remove elements each time a new value is inserted. This should still be constant-time, however adding a value with a large cost may cause multiple lower-cost values to be evicted, which will take a time proportional to the number of values affected.


# Credits

The LRUCache framework is primarily the work of [Nick Lockwood](https://github.com/nicklockwood).

([Full list of contributors](https://github.com/nicklockwood/LRUCache/graphs/contributors))
