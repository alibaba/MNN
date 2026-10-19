//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Kostub Deshmukh.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation

/**
 * An index that points to a particular character in the MTMathList. The index is a LinkedList that represents
 * a path from the beginning of the MTMathList to reach a particular atom in the list. The next node of the path
 * is represented by the subIndex. The path terminates when the subIndex is nil.
 *
 * If there is a subIndex, the subIndexType denotes what branch the path takes (i.e. superscript, subscript,
 * numerator, denominator etc.).
 * e.g in the expression 25^{2/4} the index of the character 4 is represented as:
 * (1, superscript) -> (0, denominator) -> (0, none)
 * This can be interpreted as start at index 1 (i.e. the 5) go up to the superscript.
 * Then look at index 0 (i.e. 2/4) and go to the denominator. Then look up index 0 (i.e. the 4) which this final
 * index.
 *
 * The level of an index is the number of nodes in the LinkedList to get to the final path.
 */
public class MTMathListIndex {
    
    /**
     The type of the subindex.
     
     The type of the subindex denotes what branch the path to the atom that this index points to takes.
     */
    public enum MTMathListSubIndexType: Int {
        /// The index denotes the whole atom, subIndex is nil.
        case none  = 0
        /// The position in the subindex is an index into the nucleus
        case nucleus
        /// The subindex indexes into the superscript.
        case superscript
        /// The subindex indexes into the subscript
        case ssubscript
        /// The subindex indexes into the numerator (only valid for fractions)
        case numerator
        /// The subindex indexes into the denominator (only valid for fractions)
        case denominator
        /// The subindex indexes into the radicand (only valid for radicals)
        case radicand
        /// The subindex indexes into the degree (only valid for radicals)
        case degree
    }
    
    /// The index of the associated atom.
    var atomIndex: Int
    
    /// The type of subindex, e.g. superscript, numerator etc.
    var subIndexType: MTMathListSubIndexType = .none
    
    /// The index into the sublist.
    var subIndex: MTMathListIndex?
    
    var finalIndex: Int {
        if self.subIndexType == .none {
            return self.atomIndex
        } else {
            return self.subIndex?.finalIndex ?? 0
        }
    }
    
    /// Returns the previous index if present. Returns `nil` if there is no previous index.
    func prevIndex() -> MTMathListIndex? {
        if self.subIndexType == .none {
            if self.atomIndex > 0 {
                return MTMathListIndex(level0Index: self.atomIndex - 1)
            }
        } else {
            if let prevSubIndex = self.subIndex?.prevIndex() {
                return MTMathListIndex(at: self.atomIndex, with: prevSubIndex, type: self.subIndexType)
            }
        }
        return nil
    }
    
    /// Returns the next index.
    func nextIndex() -> MTMathListIndex {
        if self.subIndexType == .none {
            return MTMathListIndex(level0Index: self.atomIndex + 1)
        } else if self.subIndexType == .nucleus {
            return MTMathListIndex(at: self.atomIndex + 1, with: self.subIndex, type: self.subIndexType)
        } else {
            return MTMathListIndex(at: self.atomIndex, with: self.subIndex?.nextIndex(), type: self.subIndexType)
        }
    }
    
    /**
     * Returns true if this index represents the beginning of a line. Note there may be multiple lines in a MTMathList,
     * e.g. a superscript or a fraction numerator. This returns true if the innermost subindex points to the beginning of a
     * line.
     */
    func isBeginningOfLine() -> Bool { self.finalIndex == 0 }
    
    func isAtSameLevel(with index: MTMathListIndex?) -> Bool {
        if self.subIndexType != index?.subIndexType {
            return false
        } else if self.subIndexType == .none {
            // No subindexes, they are at the same level.
            return true
        } else if (self.atomIndex != index?.atomIndex) {
            return false
        } else {
            return self.subIndex?.isAtSameLevel(with: index?.subIndex) ?? false
        }
    }
    
    /** Returns the type of the innermost sub index. */
    func finalSubIndexType() -> MTMathListSubIndexType {
        if self.subIndex?.subIndex != nil {
            return self.subIndex!.finalSubIndexType()
        } else {
            return self.subIndexType
        }
    }
    
    /** Returns true if any of the subIndexes of this index have the given type. */
    func hasSubIndex(ofType type: MTMathListSubIndexType) -> Bool {
        if self.subIndexType == type {
            return true
        } else {
            return self.subIndex?.hasSubIndex(ofType: type) ?? false
        }
    }
    
    func levelUp(with subIndex: MTMathListIndex?, type: MTMathListSubIndexType) -> MTMathListIndex {
        if self.subIndexType == .none {
            return MTMathListIndex(at: self.atomIndex, with: subIndex, type: type)
        }
        
        return MTMathListIndex(at: self.atomIndex, with: self.subIndex?.levelUp(with: subIndex, type: type), type: self.subIndexType)
    }
    
    func levelDown() -> MTMathListIndex? {
        if self.subIndexType == .none {
            return nil
        }
        
        if let subIndexDown = self.subIndex?.levelDown() {
            return MTMathListIndex(at: self.atomIndex, with: subIndexDown, type: self.subIndexType)
        } else {
            return MTMathListIndex(level0Index: self.atomIndex)
        }
    }
    
    /** Factory function to create a `MTMathListIndex` with no subindexes.
     @param index The index of the atom that the `MTMathListIndex` points at.
     */
    public init(level0Index: Int) {
        self.atomIndex = level0Index
    }
    
    public convenience init(at location: Int, with subIndex: MTMathListIndex?, type: MTMathListSubIndexType) {
        self.init(level0Index: location)
        self.subIndexType = type
        self.subIndex = subIndex
    }
}

extension MTMathListIndex: CustomStringConvertible {
    public var description: String {
        if self.subIndex != nil {
            return "[\(self.atomIndex), \(self.subIndexType.rawValue):\(self.subIndex!)]"
        }
        return "[\(self.atomIndex)]"
    }
}

extension MTMathListIndex: Hashable {
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.atomIndex)
        hasher.combine(self.subIndexType)
        hasher.combine(self.subIndex)
    }

}

extension MTMathListIndex: Equatable {
    public static func ==(lhs: MTMathListIndex, rhs: MTMathListIndex) -> Bool {
        if lhs.atomIndex != rhs.atomIndex || lhs.subIndexType != rhs.subIndexType {
            return false
        }
        
        if rhs.subIndex != nil {
            return rhs.subIndex == lhs.subIndex
        } else {
            return lhs.subIndex == nil
        }
    }
}
