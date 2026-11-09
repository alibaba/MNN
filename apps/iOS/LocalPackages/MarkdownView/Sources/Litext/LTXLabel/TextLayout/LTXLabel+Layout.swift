//
//  Created by Lakr233 & Helixform on 2025/2/18.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import CoreText
import Foundation
import QuartzCore

public extension LTXLabel {
    func invalidateTextLayout() {
        if let selectionRange,
           attributedText.length >= selectionRange.location + selectionRange.length
        { /* pass */ } else {
            clearSelection()
        }

        flags.layoutIsDirty = true
        setNeedsLayout()
        invalidateIntrinsicContentSize()
    }

    override var intrinsicContentSize: CGSize {
        var constraintSize = CGSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )

        if preferredMaxLayoutWidth > 0 {
            constraintSize.width = preferredMaxLayoutWidth
        } else if lastContainerSize.width > 0 {
            constraintSize.width = lastContainerSize.width
        }

        return textLayout.suggestContainerSize(
            withSize: constraintSize
        )
    }

    override func layoutSubviews() {
        super.layoutSubviews()

        let containerSize = bounds.size

        var layoutUpdateWasMade = false
        if flags.layoutIsDirty || lastContainerSize != containerSize {
            invalidateIntrinsicContentSize()
            lastContainerSize = containerSize
            textLayout.containerSize = containerSize
            flags.needsUpdateHighlightRegions = true
            flags.layoutIsDirty = false
            layoutUpdateWasMade = true
        }

        if flags.needsUpdateHighlightRegions {
            textLayout.updateHighlightRegions()
            updateAttachmentViews()
            flags.needsUpdateHighlightRegions = false
            layoutUpdateWasMade = true
        }

        if layoutUpdateWasMade {
            updateSelectionLayer()
            setNeedsDisplay()
        }
    }

    override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
        super.traitCollectionDidChange(previousTraitCollection)
        invalidateTextLayout()
    }
}
