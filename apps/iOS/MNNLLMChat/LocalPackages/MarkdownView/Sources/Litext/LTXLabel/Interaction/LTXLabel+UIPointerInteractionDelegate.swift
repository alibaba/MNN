//
//  LTXLabel+UIPointerInteractionDelegate.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/8/25.
//

import UIKit

@available(iOS 13.4, macCatalyst 13.4, *)
extension LTXLabel: UIPointerInteractionDelegate {
    public func pointerInteraction(_: UIPointerInteraction, styleFor _: UIPointerRegion) -> UIPointerStyle? {
        guard isSelectable else { return nil }
        guard parentViewController?.presentedViewController == nil else { return nil }
        return UIPointerStyle(shape: .verticalBeam(length: 1), constrainedAxes: [])
    }
}
