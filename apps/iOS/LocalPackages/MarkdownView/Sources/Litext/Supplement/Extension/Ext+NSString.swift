//
//  Ext+NSString.swift
//  Litext
//
//  Created by 秋星桥 on 3/26/25.
//

import Foundation

extension NSString {
    func rangeOfWord(at index: Int) -> NSRange {
        let options: NSString.EnumerationOptions = [.byWords, .substringNotRequired]
        var resultRange = NSRange(location: NSNotFound, length: 0)

        enumerateSubstrings(in: NSRange(location: 0, length: length), options: options) { _, substringRange, _, stop in
            if substringRange.contains(index) {
                resultRange = substringRange
                stop.pointee = true
            }
        }

        return resultRange
    }

    func rangeOfLine(at index: Int) -> NSRange {
        var startIndex = index
        while startIndex > 0, character(at: startIndex - 1) != 0x0A { // 0x0A 是换行符 '\n'
            startIndex -= 1
        }

        var endIndex = index
        while endIndex < length, character(at: endIndex) != 0x0A {
            endIndex += 1
        }

        return NSRange(location: startIndex, length: endIndex - startIndex)
    }
}
