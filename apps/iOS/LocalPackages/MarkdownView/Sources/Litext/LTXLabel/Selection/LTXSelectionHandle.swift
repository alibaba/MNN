//
//  Created by Litext Team.
//  Copyright (c) 2025 Litext Team. All rights reserved.
//

import Foundation

#if canImport(UIKit)

    protocol LTXSelectionHandleDelegate: AnyObject {
        func selectionHandleDidMove(_ type: LTXSelectionHandle.HandleType, toLocationInSuperView point: CGPoint)
    }

    public class LTXSelectionHandle: UIView {
        static let knobRadius: CGFloat = 12
        static let knobExtraResponsiveArea: CGFloat = 20

        public enum HandleType {
            case start
            case end
        }

        public let type: HandleType

        weak var delegate: LTXSelectionHandleDelegate?

        private let knobView: UIView = {
            let view = UIView()
            view.backgroundColor = .systemBlue
            view.layer.cornerRadius = knobRadius / 2
            view.layer.shadowColor = UIColor.black.cgColor
            view.layer.shadowOffset = CGSize(width: 0, height: 1)
            view.layer.shadowOpacity = 0.25
            view.layer.shadowRadius = 1.5
            return view
        }()

        private let stickView: UIView = {
            let view = UIView()
            view.backgroundColor = .systemBlue
            return view
        }()

        public init(type: HandleType) {
            self.type = type
            super.init(frame: .zero)
            setupView()
        }

        required init?(coder: NSCoder) {
            type = .start
            super.init(coder: coder)
            setupView()
        }

        private func setupView() {
            backgroundColor = .clear
            isUserInteractionEnabled = true
            addSubview(stickView)
            addSubview(knobView)
            let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panGesture.cancelsTouchesInView = true
            addGestureRecognizer(panGesture)
        }

        override public func layoutSubviews() {
            super.layoutSubviews()
            let stickWidth = 2
            stickView.frame = .init(
                x: bounds.midX - CGFloat(stickWidth) / 2,
                y: bounds.minY,
                width: CGFloat(stickWidth),
                height: bounds.height
            )

            let knobRadius: CGFloat = knobView.layer.cornerRadius
            switch type {
            case .start:
                knobView.frame = .init(
                    x: bounds.midX - knobRadius,
                    y: 0,
                    width: knobRadius * 2,
                    height: knobRadius * 2
                )
            case .end:
                knobView.frame = .init(
                    x: bounds.midX - knobRadius,
                    y: bounds.height - knobRadius * 2,
                    width: knobRadius * 2,
                    height: knobRadius * 2
                )
            }
        }

        private var frameAtGestureBegin: CGRect = .zero

        @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
            switch gesture.state {
            case .began:
                frameAtGestureBegin = frame
                fallthrough
            case .changed:
                let translation = gesture.translation(in: superview)
                let newFrame = CGRect(
                    x: frameAtGestureBegin.origin.x + translation.x,
                    y: frameAtGestureBegin.origin.y + translation.y,
                    width: frameAtGestureBegin.width,
                    height: frameAtGestureBegin.height
                )
                delegate?.selectionHandleDidMove(type, toLocationInSuperView: .init(x: newFrame.midX, y: newFrame.midY))
            default: return
            }
        }

        override public func point(inside point: CGPoint, with _: UIEvent?) -> Bool {
            let touchRect = bounds.insetBy(
                dx: -Self.knobExtraResponsiveArea,
                dy: -Self.knobExtraResponsiveArea
            )
            return touchRect.contains(point)
        }
    }
#endif
