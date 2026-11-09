# Swift Collections

**Swift Collections** is an open-source package of data structure implementations for the Swift programming language.

Read more about the package, and the intent behind it, in the [announcement on swift.org][announcement].

[announcement]: https://swift.org/blog/swift-collections

## Contents

The package currently provides the following implementations:

- [`BitSet`][BitSet] and [`BitArray`][BitArray], dynamic bit collections.

- [`Deque<Element>`][Deque], a double-ended queue backed by a ring buffer. Deques are range-replaceable, mutable, random-access collections.

- [`Heap`][Heap], a min-max heap backed by an array, suitable for use as a priority queue.

- [`OrderedSet<Element>`][OrderedSet], a variant of the standard `Set` where the order of items is well-defined and items can be arbitrarily reordered. Uses a `ContiguousArray` as its backing store, augmented by a separate hash table of bit packed offsets into it.

- [`OrderedDictionary<Key, Value>`][OrderedDictionary], an ordered variant of the standard `Dictionary`, providing similar benefits.

- [`TreeSet`][TreeSet] and [`TreeDictionary`][TreeDictionary], persistent hashed collections implementing Compressed Hash-Array Mapped Prefix Trees (CHAMP). These work similar to the standard `Set` and `Dictionary`, but they excel at use cases that mutate shared copies, offering dramatic memory savings and radical time improvements.  

[BitSet]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/bitcollections/bitset
[BitArray]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/bitcollections/bitarray
[Deque]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/dequemodule/deque
[Heap]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/heapmodule/heap
[OrderedSet]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/orderedcollections/orderedset
[OrderedDictionary]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/orderedcollections/ordereddictionary
[TreeSet]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/hashtreecollections/treeset
[TreeDictionary]: https://swiftpackageindex.com/apple/swift-collections/1.1.0/documentation/hashtreecollections/treedictionary

Swift Collections uses the same modularization approach as [**Swift Numerics**](https://github.com/apple/swift-numerics): it provides a standalone module for each thematic group of data structures it implements. For instance, if you only need a double-ended queue type, you can pull in only that by importing `DequeModule`. `OrderedSet` and `OrderedDictionary` share much of the same underlying implementation, so they are provided by a single module, called `OrderedCollections`. However, there is also a top-level `Collections` module that gives you every collection type with a single import statement:

``` swift
import Collections

var deque: Deque<String> = ["Ted", "Rebecca"]
deque.prepend("Keeley")
deque.append("Nathan")
print(deque) // ["Keeley", "Ted", "Rebecca", "Nathan"]
```

## Project Status

The Swift Collections package is source stable. The version numbers follow [Semantic Versioning][semver] -- source breaking changes to public API can only land in a new major version.

[semver]: https://semver.org

### Public API 

The public API of version 1.2 of the `swift-collections` package consists of non-underscored declarations that are marked `public` in the `Collections`, `BitCollections`, `DequeModule`, `HeapModule`, `OrderedCollections` and `HashTreeCollections` modules. 

Interfaces that aren't part of the public API may continue to change in any release, including patch releases.

By "underscored declarations" we mean declarations that have a leading underscore anywhere in their fully qualified name. For instance, here are some names that wouldn't be considered part of the public API, even if they were technically marked public:

- `FooModule.Bar._someMember(value:)` (underscored member)
- `FooModule._Bar.someMember` (underscored type)
- `_FooModule.Bar` (underscored module)
- `FooModule.Bar.init(_value:)` (underscored initializer)

If you have a use case that requires using underscored (or otherwise non-public) APIs, please [submit a Feature Request][enhancement] describing it! We'd like the public interface to be as useful as possible -- although preferably without compromising safety or limiting future evolution.

This source compatibility promise only applies to swift-collection when built as a Swift package. (The repository also contains unstable configurations for building swift-collections using CMake and Xcode. These configurations are provided for internal Swift project use only -- such as for building the (private) swift-collections binaries that ship within Swift toolchains.)

Note that the files in the `Tests`, `Utils`, `Documentation`, `Xcode`, `cmake` and `Benchmarks` subdirectories may change at whim; they may be added, modified or removed in any new release. Do not rely on anything about them.

Future minor versions of the package may update these rules as needed.

### Minimum Required Swift Toolchain Version

We'd like this package to quickly embrace Swift language and toolchain improvements that are relevant to its mandate. Accordingly, from time to time, new versions of this package require clients to upgrade to a more recent Swift toolchain release. (This allows the package to make use of new language/stdlib features, build on compiler bug fixes, and adopt new package manager functionality as soon as they are available.) Patch (i.e., bugfix) releases will not increase the required toolchain version, but any minor (i.e., new feature) release may do so.

The following table maps package releases to their minimum required Swift toolchain release:

| Package version         | Swift version   | Xcode release |
| ----------------------- | --------------- | ------------- |
| swift-collections 1.0.x | >= Swift 5.3.2  | >= Xcode 12.4 |
| swift-collections 1.1.x | >= Swift 5.7.2  | >= Xcode 14.2 |
| swift-collections 1.2.x | >= Swift 5.10.0 | >= Xcode 15.3 |

(Note: the package has no minimum deployment target, so while it does require clients to use a recent Swift toolchain to build it, the code itself is able to run on any OS release that supports running Swift code.)

## Using **Swift Collections** in your project

To use this package in a SwiftPM project, you need to set it up as a package dependency:

```swift
// swift-tools-version:6.1
import PackageDescription

let package = Package(
  name: "MyPackage",
  dependencies: [
    .package(
      url: "https://github.com/apple/swift-collections.git", 
      .upToNextMinor(from: "1.2.0") // or `.upToNextMajor
    )
  ],
  targets: [
    .target(
      name: "MyTarget",
      dependencies: [
        .product(name: "Collections", package: "swift-collections")
      ]
    )
  ]
)
```

## Contributing to Swift Collections

We have a dedicated [Swift Collections Forum][forum] where people can ask and answer questions on how to use or work on this package. It's also a great place to discuss its evolution.

[forum]: https://forums.swift.org/c/related-projects/collections

If you find something that looks like a bug, please open a [Bug Report][bugreport]! Fill out as many details as you can.

### Branching Strategy

We maintain separate branches for each minor version of the package:

| Package version         | Branch      | Status   |
| ----------------------- | ----------- | -------- |
| swift-collections 1.0.x | release/1.0 | Obsolete |
| swift-collections 1.1.x | release/1.1 | Critical bugfixes only | 
| swift-collections 1.2.x | release/1.2 | Critical bugfixes only |
| swift-collections 1.3.x | main        | Feature work towards next minor release |
| n.a.                    | future      | Experimental prototyping |

Changes must land on the branch corresponding to the earliest release that they will need to ship on. They are periodically propagated to subsequent branches, in the following direction:

`release/1.0` → `release/1.1` → `release/1.2` → `main`

For example, anything landing on `release/1.1` will eventually appear on `release/1.2` and then `main` too; there is no need to file standalone PRs for each release line. Change propagation is not instantaneous, as it currently requires manual work -- it is performed by project maintainers.

### Working on the package

We have some basic [documentation on package internals](./Documentation/Internals/README.md) that will help you get started.

By submitting a pull request, you represent that you have the right to license your contribution to Apple and the community, and agree by submitting the patch that your contributions are licensed under the [Swift License](https://swift.org/LICENSE.txt), a copy of which is [provided in this repository](LICENSE.txt).

#### Fixing a bug or making a small improvement

1. Make sure to start by checking out the appropriate branch for the minor release you want the fix to ship in. (See above.)
2. [Submit a PR][PR] with your change. If there is an [existing issue][issues] for the bug you're fixing, please include a reference to it.
3. Make sure to add tests covering whatever changes you are making.

[PR]: https://github.com/apple/swift-collections/compare
[issues]: https://github.com/apple/swift-collections/issues

[bugreport]: https://github.com/apple/swift-collections/issues/new?assignees=&labels=bug&template=BUG_REPORT.md

#### Proposing a small enhancement

1. Raise a [Feature Request][enhancement]. Discuss why it would be important to implement it.
2. Submit a PR with your implementation, participate in the review discussion.
3. When there is a consensus that the feature is desirable, and the implementation works well, it is fully tested and documented, then it will be merged. 
4. Rejoice!

[enhancement]: https://github.com/apple/swift-collections/issues/new?assignees=&labels=enhancement&template=FEATURE_REQUEST.md

#### Proposing the addition of a new data structure

**Note:** As of 2024, we are fully preoccupied with refactoring our existing data structures to support noncopyable and/or nonescapable element types; this includes designing new container protocols around them. I don't expect we'll have capacity to work on any major new data structure implementations until this effort is complete.

<!--

We intend this package to collect generally useful data structures -- the ones that ought to be within easy reach of every Swift engineer's basic toolbox. The implementations we ship need to be of the highest technical quality, polished to the same shine as anything that gets included in the Swift Standard Library. (The only real differences are that this package isn't under the formal Swift Evolution process, and its code isn't ABI stable.) 

Accordingly, adding a new data structure to this package is not an easy or quick process, and not all useful data structures are going to be a good fit. 

If you have an idea for a data structure that might make a good addition to this package, please start a topic on the [forum], explaining why you believe it would be important to implement it. This way we can figure out if it would be right for the package, discuss implementation strategies, and plan to allocate capacity to help.

Not all data structures will reach a high enough level of usefulness to ship in this package -- those that have a more limited audience might work better as a standalone package. Of course, reasonable people might disagree on the importance of including any particular data structure; but at the end of the day, the decision whether to take an implementation is up to the maintainers of this package.

If maintainers have agreed that your implementation would likely make a good addition, then it's time to start work on it. Submit a PR with your implementation as soon as you have something that's ready to show! We'd love to get involved as early as you like. Historically, the best additions resulted from close work between the contributor and a package maintainer.

Participate in the review discussion, and adapt code accordingly. Sometimes we may need to go through several revisions over multiple months! This is fine -- it makes the end result that much better. When there is a consensus that the feature is ready, and the implementation is fully tested and documented, the PR will be merged by a maintainer. This is good time for a small celebration -- merging is a good indicator that the addition will ship at some point.

Historically, PRs adding a new data structure have typically been merged to a new feature branch rather than directly to a release branch or `main`, and there was an extended amount of time between the initial merge and the tag that shipped the new feature. Nobody likes to wait, but getting a new data structure implementation from a state that was ready to merge to a state that's ready to ship is actually quite difficult work, and it takes maintainer time and effort that needs to be scheduled in advance. The closer an implementation is to the coding conventions and performance baseline of the Standard Library, the shorter this wait is likely to become, and the fewer changes there will be between merging and shipping.

-->

### Code of Conduct

Like all Swift.org projects, we would like the Swift Collections project to foster a diverse and friendly community. We expect contributors to adhere to the [Swift.org Code of Conduct](https://swift.org/code-of-conduct/). A copy of this document is [available in this repository][coc].

[coc]: CODE_OF_CONDUCT.md

### Contact information

The current code owner of this package is Karoy Lorentey ([@lorentey](https://github.com/lorentey)). You can contact him [on the Swift forums](https://forums.swift.org/u/lorentey/summary), or by writing an email to klorentey at apple dot com. (Please keep it related to this project.)

In case of moderation issues, you can also directly contact a member of the [Swift Core Team](https://swift.org/community/#community-structure).
