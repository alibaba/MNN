//
//  Created by Lakr233 & Helixform on 2025/2/18.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import Foundation

extension LTXLabel {
    func isLocationAboveAttachmentView(location: CGPoint) -> Bool {
        for view in attachmentViews {
            if view.frame.contains(location) {
                return true
            }
        }
        return false
    }

    func updateAttachmentViews() {
        let viewsToRemove = attachmentViews
        var newAttachmentViews: Set<LTXPlatformView> = []

        for highlightRegion in highlightRegions {
            guard let attachment = highlightRegion.attributes[LTXAttachmentAttributeName] as? LTXAttachment,
                  let view = attachment.view else { continue }

            if view.superview == self {
                newAttachmentViews.insert(view)
            } else {
                addSubview(view)
                newAttachmentViews.insert(view)
            }

            let rect = highlightRegion.rects.first!.cgRectValue
            let convertedRect = convertRectFromTextLayout(rect, insetForInteraction: false)
            view.frame = convertedRect
        }

        for view in viewsToRemove {
            if !newAttachmentViews.contains(view) {
                view.removeFromSuperview()
            }
        }

        attachmentViews = newAttachmentViews
    }
}
