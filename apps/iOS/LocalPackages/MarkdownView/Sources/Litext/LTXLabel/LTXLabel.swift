//
//  Created by Lakr233 & Helixform on 2025/2/18.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import CoreFoundation
import CoreText
import Foundation
import QuartzCore

public class LTXLabel: LTXPlatformView, Identifiable {
    public let id: UUID = .init()

    // MARK: - Public Properties

    public var attributedText: NSAttributedString = .init() {
        didSet { textLayout = LTXTextLayout(attributedString: attributedText) }
    }

    public var preferredMaxLayoutWidth: CGFloat = 0 {
        didSet {
            if preferredMaxLayoutWidth != oldValue {
                invalidateTextLayout()
            }
        }
    }

    override public var frame: CGRect {
        get { super.frame }
        set {
            guard newValue != super.frame else { return }
            super.frame = newValue
            invalidateTextLayout()
        }
    }

    public var isSelectable: Bool = false {
        didSet { if !isSelectable { clearSelection() } }
    }

    public internal(set) var isInteractionInProgress = false

    public weak var delegate: LTXLabelDelegate?

    // MARK: - Internal Properties

    var textLayout: LTXTextLayout = .init(attributedString: .init()) {
        didSet { invalidateTextLayout() }
    }

    var attachmentViews: Set<LTXPlatformView> = []
    var highlightRegions: [LTXHighlightRegion] { textLayout.highlightRegions }
    var activeHighlightRegion: LTXHighlightRegion?
    var lastContainerSize: CGSize = .zero

    public internal(set) var selectionRange: NSRange? {
        didSet {
            updateSelectionLayer()
            if selectionRange != oldValue {
                delegate?.ltxLabelSelectionDidChange(self, selection: selectionRange)
            }
        }
    }

    var selectedLinkForMenuAction: URL?
    var selectionLayer: CAShapeLayer?

    #if canImport(UIKit) && !targetEnvironment(macCatalyst)
        var selectionHandleStart: LTXSelectionHandle = .init(type: .start)
        var selectionHandleEnd: LTXSelectionHandle = .init(type: .end)
    #endif

    var interactionState = InteractionState()
    var flags = Flags()

    // MARK: - Initialization

    override public init(frame: CGRect) {
        super.init(frame: frame)
        registerNotificationCenterForSelectionDeduplicate()

        backgroundColor = .clear
        installContextMenuInteraction()
        installTextPointerInteraction()

        isMultipleTouchEnabled = false
        isExclusiveTouch = true

        #if targetEnvironment(macCatalyst)
        #else
            clipsToBounds = false // for selection handle
            selectionHandleStart.isHidden = true
            selectionHandleStart.delegate = self
            addSubview(selectionHandleStart)
            selectionHandleEnd.isHidden = true
            selectionHandleEnd.delegate = self
            addSubview(selectionHandleEnd)
        #endif
    }

    @available(*, unavailable)
    public required init?(coder _: NSCoder) {
        fatalError()
    }

    deinit {
        attributedText = .init()
        attachmentViews = []
        clearSelection()
        deactivateHighlightRegion()
        NotificationCenter.default.removeObserver(self)
    }

    override public func didMoveToWindow() {
        super.didMoveToWindow()
        clearSelection()
        invalidateTextLayout()
    }
}

extension LTXLabel {
    struct InteractionState {
        var initialTouchLocation: CGPoint = .zero
        var clickCount: Int = 1
        var lastClickTime: TimeInterval = 0
        var isFirstMove: Bool = false
    }

    struct Flags {
        var layoutIsDirty: Bool = false
        var needsUpdateHighlightRegions: Bool = false
    }
}
