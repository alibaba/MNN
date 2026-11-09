//===----------------------------------------------------------------------===//
//
// This source file is part of the Swift Collections open source project
//
// Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

/// A type with manually controlled hash values, for easy testing of collision
/// scenarios.
struct Collider:
  Hashable, CustomStringConvertible, CustomDebugStringConvertible
{
  var identity: Int
  var hash: Hash

  init(_ identity: Int, _ hash: Hash) {
    self.identity = identity
    self.hash = hash
  }

  init(_ identity: Int) {
    self.identity = identity
    self.hash = Hash(identity)
  }

  init(_ identity: String) {
    self.hash = Hash(identity)!
    self.identity = hash.value
  }

  static func ==(left: Self, right: Self) -> Bool {
    guard left.identity == right.identity else { return false }
    precondition(left.hash == right.hash)
    return true
  }

  func hash(into hasher: inout Hasher) {
    hasher.combine(hash.value)
  }

  var description: String {
    "\(identity)(#\(hash))"
  }

  var debugDescription: String {
    description
  }
}

/// A type with precisely controlled hash values. This can be used to set up
/// hash tables with fully deterministic contents.
struct RawCollider:
  Hashable, CustomStringConvertible
{
  var identity: Int
  var hash: Hash

  init(_ identity: Int, _ hash: Hash) {
    self.identity = identity
    self.hash = hash
  }

  init(_ identity: Int) {
    self.identity = identity
    self.hash = Hash(identity)
  }

  init(_ identity: String) {
    self.hash = Hash(identity)!
    self.identity = hash.value
  }

  static func ==(left: Self, right: Self) -> Bool {
    guard left.identity == right.identity else { return false }
    precondition(left.hash == right.hash)
    return true
  }

  var hashValue: Int {
    fatalError("Don't")
  }

  func hash(into hasher: inout Hasher) {
    fatalError("Don't")
  }

  func _rawHashValue(seed: Int) -> Int {
    hash.value
  }

  var description: String {
    "\(identity)#\(hash)"
  }
}
