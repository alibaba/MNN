//
//  LTXLabel+Delegate.swift
//  Litext
//
//  Created by 秋星桥 on 7/5/25.
//

import Foundation

public protocol LTXLabelDelegate: AnyObject {
    func ltxLabelDidTapOnHighlightContent(
        _ ltxLabel: LTXLabel,
        region: LTXHighlightRegion?,
        location: CGPoint
    )

    func ltxLabelSelectionDidChange(
        _ ltxLabel: LTXLabel,
        selection: NSRange?
    )

    // useful for moving scrollview accordingly to handle selection
    func ltxLabelDetectedUserEventMovingAtLocation(
        _ ltxLabel: LTXLabel,
        location: CGPoint
    )
}
