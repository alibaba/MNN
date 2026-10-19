//
//  LTXAttachment+Extension.swift
//  MarkdownView
//
//  Created by 秋星桥 on 3/27/25.
//

import Foundation
import Litext

private class LTXHolderAttachment: LTXAttachment {
    let attrString: NSAttributedString
    init(attrString: NSAttributedString) {
        self.attrString = attrString
        super.init()
    }

    override func attributedStringRepresentation() -> NSAttributedString {
        attrString
    }
}

extension LTXAttachment {
    static func hold(attrString: NSAttributedString) -> LTXAttachment {
        LTXHolderAttachment(attrString: attrString)
    }
}
