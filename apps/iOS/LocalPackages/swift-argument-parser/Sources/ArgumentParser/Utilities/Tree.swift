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

final class Tree<Element> {
  var element: Element
  weak var parent: Tree?
  var children: [Tree]
  
  var isRoot: Bool { parent == nil }
  var isLeaf: Bool { children.isEmpty }
  var hasChildren: Bool { !isLeaf }
  
  init(_ element: Element) {
    self.element = element
    self.parent = nil
    self.children = []
  }
  
  func addChild(_ tree: Tree) {
    children.append(tree)
    tree.parent = self
  }
}

extension Tree: Hashable {
  static func == (lhs: Tree<Element>, rhs: Tree<Element>) -> Bool {
    lhs === rhs
  }
  
  func hash(into hasher: inout Hasher) {
    hasher.combine(ObjectIdentifier(self))
  }
}

extension Tree {
  /// Returns a path of tree nodes that traverses from this node to the first
  /// node (breadth-first) that matches the given predicate.
  func path(toFirstWhere predicate: (Element) -> Bool) -> [Tree] {
    var visited: Set<Tree> = []
    var toVisit: [Tree] = [self]
    var currentIndex = 0
    
    // For each node, the neighbor that is most efficiently used to reach
    // that node.
    var cameFrom: [Tree: Tree] = [:]
    
    while let current = toVisit[currentIndex...].first {
      currentIndex += 1
      if predicate(current.element) {
        // Reconstruct the path from `self` to `current`.
        return sequence(first: current, next: { cameFrom[$0] }).reversed()
      }
      visited.insert(current)
      
      for child in current.children where !visited.contains(child) {
        if !toVisit.contains(child) {
          toVisit.append(child)
        }
        
        // Coming from `current` is the best path to `neighbor`.
        cameFrom[child] = current
      }
    }
    
    // Didn't find a path!
    return []
  }
}

extension Tree where Element == ParsableCommand.Type {
  func path(to element: Element) -> [Element] {
    path(toFirstWhere: { $0 == element }).map { $0.element }
  }
  
  func firstChild(equalTo element: Element) -> Tree? {
    children.first(where: { $0.element == element })
  }
  
  func firstChild(withName name: String) -> Tree? {
      children.first(where: {
          $0.element._commandName == name || $0.element.configuration.aliases.contains(name)
      })
  }
  
  convenience init(root command: ParsableCommand.Type) throws {
    self.init(command)
    for subcommand in command.configuration.subcommands {
      if subcommand == command {
        throw InitializationError.recursiveSubcommand(subcommand)
      }
      // We don't allow an alias that has the same name as the command itself.
      if subcommand.configuration.aliases.contains(subcommand._commandName) {
        throw InitializationError.aliasMatchingCommand(subcommand)
      }
      try addChild(Tree(root: subcommand))
    }
  }
    
  enum InitializationError: Error {
    case recursiveSubcommand(ParsableCommand.Type)
    case aliasMatchingCommand(ParsableCommand.Type)
  }
}
