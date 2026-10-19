//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

import XCTest
@testable import ArgumentParser

final class TreeTests: XCTestCase {
}

// MARK: -

let tree: Tree<Int> = {
  let tree = Tree(1)
  for x in 11...13 {
    let node = Tree(x)
    tree.addChild(node)
    for y in 1...3 {
      let subnode = Tree(x * 10 + y)
      node.addChild(subnode)
    }
  }
  return tree
}()

extension TreeTests {
  func testHierarchy() {
    XCTAssertEqual(tree.element, 1)
    XCTAssertEqual(tree.children.map { $0.element }, [11, 12, 13])
    XCTAssertEqual(
      tree.children.flatMap { $0.children.map { $0.element } },
      [111, 112, 113, 121, 122, 123, 131, 132, 133])
  }
  
  func testSearch() {
    XCTAssertEqual(
      tree.path(toFirstWhere: { $0 == 1 }).map { $0.element },
      [1])
    XCTAssertEqual(
      tree.path(toFirstWhere: { $0 == 13 }).map { $0.element },
      [1, 13])
    XCTAssertEqual(
      tree.path(toFirstWhere: { $0 == 133 }).map { $0.element },
      [1, 13, 133])
    
    XCTAssertTrue(tree.path(toFirstWhere: { $0 < 0 }).isEmpty)
  }
}

extension TreeTests {
  struct A: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [A.self])
  }
  struct Root: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [Sub.self])
  }
  struct Sub: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [Sub.self])
  }

  struct RootWithNamedNestedSub: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [NestedSub.self])

    struct NestedSub: ParsableCommand {
      static let configuration = CommandConfiguration(commandName: "sub", aliases: ["sub"])
    }
  }
    
  struct RootWithNestedSub: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [NestedSub.self])

    struct NestedSub: ParsableCommand {
      static let configuration = CommandConfiguration(aliases: ["nested-sub"])
    }
  }

  func testInitializationWithRecursiveSubcommand() {
    XCTAssertThrowsError(try Tree(root: A.asCommand))
    XCTAssertThrowsError(try Tree(root: Root.asCommand))
  }

  func testInitializationWithMatchingAliases() {
    XCTAssertThrowsError(try Tree(root: RootWithNamedNestedSub.asCommand))
    XCTAssertThrowsError(try Tree(root: RootWithNestedSub.asCommand))
  }
}
