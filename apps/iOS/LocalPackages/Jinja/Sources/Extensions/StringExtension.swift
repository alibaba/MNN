//
//  StringExtension.swift
//
//
//  Created by John Mai on 2024/3/20.
//

import Foundation

extension String {
    subscript(i: Int) -> Character {
        self[index(startIndex, offsetBy: i)]
    }

    func slice(start: Int, end: Int) -> Self {
        let startPosition = index(startIndex, offsetBy: start)
        let endPosition = index(startIndex, offsetBy: end)
        return String(self[startPosition ..< endPosition])
    }

    func replacingOccurrences(of target: String, with replacement: String, count: Int) -> String {
        guard count > 0 else { return self }

        var result = self
        var replacementCount = 0
        var searchStartIndex = result.startIndex

        while replacementCount < count,
            let range = result.range(of: target, range: searchStartIndex ..< result.endIndex)
        {
            result.replaceSubrange(range, with: replacement)
            replacementCount += 1

            let offset = replacement.count
            searchStartIndex = result.index(range.lowerBound, offsetBy: offset)
        }

        return result
    }
}
