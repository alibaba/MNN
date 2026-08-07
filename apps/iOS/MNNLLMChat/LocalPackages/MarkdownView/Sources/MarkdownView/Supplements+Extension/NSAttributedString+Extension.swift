//
//  NSAttributedString+Extension.swift
//  MarkdownView
//
//  Created by 秋星桥 on 1/23/25.
//

import CoreText
import Foundation
import Litext

public extension NSAttributedString.Key {
    @inline(__always) static let coreTextRunDelegate = NSAttributedString.Key(rawValue: kCTRunDelegateAttributeName as String)
}
