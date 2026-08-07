# Change Log

## [1.2.1](https://github.com/nicklockwood/LRUCache/releases/tag/1.2.1) (2026-01-11)

- Added optional `clearsOnMemoryPressure` parameter to opt-out of automatic cache clearing behavior

## [1.2.0](https://github.com/nicklockwood/LRUCache/releases/tag/1.2.0) (2025-10-17)

- DispatchSourceMemoryPressure is now used to clear cache automatically on all Apple platforms
- Deprecated `LRUCacheMemoryWarningNotification`

## [1.1.2](https://github.com/nicklockwood/LRUCache/releases/tag/1.1.2) (2025-08-05)

- Set correct minimum OS versions

## [1.1.1](https://github.com/nicklockwood/LRUCache/releases/tag/1.1.1) (2025-08-02)

- Improved thread safety
- Bumped tools version in Package.swift

## [1.1.0](https://github.com/nicklockwood/LRUCache/releases/tag/1.1.0) (2025-08-02)

- Added Sendable conformance
- Setting `totalCostLimit` and `countLimit` is now thread-safe
- Added `hasValue(forKey:)` method to check for the presence of a value without bumping its priority
- Renamed `removeAllValues()` to `removeAll()`, `allKeys` to `orderedKeys` and `allValues` to `orderedValues`
- Improved the performance of `allKeys` and `allValues` getters (now called `orderedKeys`/`orderedValues`)
- Added new, much faster unordered `keys` and `values` accessors
- Bumped required Swift version to 5.7

## [1.0.7](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.7) (2024-01-26)

- Fixed watchOS compatibility issue

## [1.0.6](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.6) (2024-01-22)

- Fixed bug where `totalCost` wasn't reset when calling `removeAllValues()`
- Automatic cache clearing in the event of a memory warning now works on visionOS

## [1.0.5](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.5) (2024-01-20)

- Added `allKeys` computed property

## [1.0.4](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.4) (2022-09-16)

- Added `allValues` computed property
- Added watchOS compatibility
- Fixed warnings on Xcode 14

## [1.0.3](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.3) (2021-08-08)

- Fixed warnings on Xcode 13.3

## [1.0.2](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.2) (2021-08-08)

- Significantly improved performance when cache is full by using a double-linked list

## [1.0.1](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.1) (2021-08-06)

- Fixed a small memory leak on iOS, where `NSNotification` observers weren't released
- Added `LRUCacheMemoryWarningNotification` for simpler cross-platform testing

## [1.0.0](https://github.com/nicklockwood/LRUCache/releases/tag/1.0.0) (2021-08-05)

- First release
