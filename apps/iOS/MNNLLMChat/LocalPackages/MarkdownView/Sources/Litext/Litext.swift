//
//  Litext.swift
//  Litext
//
//  Created by 秋星桥 on 3/27/25.
//

import Foundation

public let LTXReplacementText = "\u{FFFC}"
public let LTXAttachmentAttributeName = NSAttributedString.Key("LTXAttachment")
public let LTXLineDrawingCallbackName = NSAttributedString.Key("LTXLineDrawingCallback")

public extension NSAttributedString.Key {
    @inline(__always) static let ltxAttachment = LTXAttachmentAttributeName
    @inline(__always) static let ltxLineDrawingCallback = LTXLineDrawingCallbackName
}
