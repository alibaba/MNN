//
//  FloatingButton.swift
//  FloatingButton
//
//  Created by Alisa Mylnikova on 27/11/2019.
//  Copyright © 2019 Exyte. All rights reserved.
//

import SwiftUI

public enum Direction {
    case left, right, top, bottom
}

public enum LayoutDirection {
    case clockwise, counterClockwise
}

public enum Alignment {
    case left, right, top, bottom, center
}

public struct FloatingButton<MainView, ButtonView>: View where MainView: View, ButtonView: View {

    fileprivate enum MenuType {
        case straight
        case circle
    }
    
    fileprivate var mainButtonView: MainView
    fileprivate var buttons: [SubmenuButton<ButtonView>]
    
    fileprivate var menuType: MenuType = .straight
    fileprivate var spacing: CGFloat = 10
    fileprivate var initialScaling: CGFloat = 1
    fileprivate var initialOffset: CGPoint = CGPoint()
    fileprivate var initialOpacity: Double = 1
    fileprivate var animation: Animation = .easeInOut(duration: 0.4)
    fileprivate var delays: [Double] = []
    fileprivate var mainZStackAlignment: SwiftUI.Alignment = .center
    fileprivate var inverseZIndex: Bool = false

    fileprivate var wholeMenuSize: Binding<CGSize> = .constant(.zero)
    fileprivate var menuButtonsSize: Binding<CGSize> = .constant(.zero)
    
    // straight
    fileprivate var direction: Direction = .left
    fileprivate var alignment: Alignment = .center
    
    // circle
    fileprivate var layoutDirection: LayoutDirection = .clockwise
    fileprivate var startAngle: Double = .pi
    fileprivate var endAngle: Double = 2 * .pi
    fileprivate var radius: Double?
    
    @State private var privateIsOpen: Bool = false
    var isOpenBinding: Binding<Bool>?
    var isOpen: Bool {
        get { isOpenBinding?.wrappedValue ?? privateIsOpen }
    }
    
    @State private var coords: [CGPoint] = []
    @State private var alignmentOffsets: [CGSize] = []
    @State private var initialPositions: [CGPoint] = [] // if there is initial offset
    @State private var sizes: [CGSize] = []
    @State private var mainButtonSize = CGSize()
    
    private init(mainButtonView: MainView, buttons: [SubmenuButton<ButtonView>], isOpenBinding: Binding<Bool>?) {
        self.mainButtonView = mainButtonView
        self.buttons = buttons
        self.isOpenBinding = isOpenBinding
    }
    
    public init(mainButtonView: MainView, buttons: [ButtonView]) {
        self.mainButtonView = mainButtonView
        self.buttons = buttons.map { SubmenuButton(buttonView: $0) }
    }
    
    public init(mainButtonView: MainView, buttons: [ButtonView], isOpen: Binding<Bool>) {
        self.mainButtonView = mainButtonView
        self.buttons = buttons.map { SubmenuButton(buttonView: $0) }
        self.isOpenBinding = isOpen
    }
    
    public var body: some View {
        ZStack(alignment: mainZStackAlignment) {
            ForEach((0..<buttons.count), id: \.self) { i in
                buttons[i]
                    .background(SubmenuButtonPreferenceViewSetter())
                    .offset(alignmentOffsets.isEmpty ? .zero : alignmentOffsets[i])
                    .offset(buttonOffset(at: i))
                    .scaleEffect(isOpen ? 1 : initialScaling)
                    .opacity(mainButtonSize == .zero ? 0 : isOpen ? 1 : initialOpacity)
                    .animation(buttonAnimation(at: i), value: isOpen)
                    .zIndex(Double(inverseZIndex ? (buttons.count - i - 1) : 0))
            }

            MainButtonViewInternal(isOpen: isOpenBinding ?? $privateIsOpen, mainButtonView: mainButtonView)
                .buttonStyle(PlainButtonStyle())
                .sizeGetter($mainButtonSize)
                .zIndex(Double(inverseZIndex ? buttons.count : 1))
        }
        .onPreferenceChange(SubmenuButtonPreferenceKey.self) { (sizes) in
            let sizes = sizes.map { CGSize(width: CGFloat(Int($0.width + 0.5)), height: CGFloat(Int($0.height + 0.5))) }
            DispatchQueue.main.async {
                if sizes != self.sizes {
                    self.sizes = sizes
                    calculateCoords()
                }
            }
        }
        .onChange(of: mainButtonSize) { _ in
            calculateCoords()
        }
    }
    
    fileprivate func buttonOffset(at i: Int) -> CGSize {
        isOpen
        ? CGSize(width: coords[safe: i].x, height: coords[safe: i].y)
        : CGSize(width: (initialPositions.isEmpty ? 0 : initialPositions[safe: i].x),
                 height: (initialPositions.isEmpty ? 0 : initialPositions[safe: i].y))
    }
    
    fileprivate func buttonAnimation(at i: Int) -> Animation {
        animation.delay(delays.isEmpty ? Double(0) :
                            (isOpen ? delays[delays.count - i - 1] : delays[i]))
    }
    
    fileprivate func calculateCoords() {
        switch menuType {
        case .straight:
            calculateCoordsStraight()
        case .circle:
            calculateCoordsCircle()
        }
    }
    
    fileprivate func calculateCoordsStraight() {
        guard sizes.count > 0, mainButtonSize != .zero else {
            return
        }

        let sizes = sizes.map { roundToTwoDigits($0) }
        let allSizes = [roundToTwoDigits(mainButtonSize)] + sizes
        
        var coord = CGPoint.zero
        coords = (0..<sizes.count).map { i -> CGPoint in
            let width = allSizes[i].width / 2 + allSizes[i+1].width / 2
            let height = allSizes[i].height / 2 + allSizes[i+1].height / 2

            switch direction {
            case .left:
                coord = CGPoint(x: coord.x - width - spacing, y: coord.y)
            case .right:
                coord = CGPoint(x: coord.x + width + spacing, y: coord.y)
            case .top:
                coord = CGPoint(x: coord.x, y: coord.y - height - spacing)
            case .bottom:
                coord = CGPoint(x: coord.x, y: coord.y + height + spacing)
            }
            return coord
        }
        
        if initialOffset.x != 0 || initialOffset.y != 0 {
            initialPositions = (0..<sizes.count).map { i -> CGPoint in
                CGPoint(x: coords[i].x + initialOffset.x,
                        y: coords[i].y + initialOffset.y)
            }
        } else {
            initialPositions = Array(repeating: .zero, count: sizes.count)
        }
        
        alignmentOffsets = (0..<sizes.count).map { i -> CGSize in
            switch alignment {
            case .left:
                return CGSize(width: sizes[i].width / 2 - mainButtonSize.width / 2, height: 0)
            case .right:
                return CGSize(width: -sizes[i].width / 2 + mainButtonSize.width / 2, height: 0)
            case .top:
                return CGSize(width: 0, height: sizes[i].height / 2 - mainButtonSize.height / 2)
            case .bottom:
                return CGSize(width: 0, height: -sizes[i].height / 2 + mainButtonSize.height / 2)
            case .center:
                return CGSize()
            }
        }

        var buttonsSize = CGSize.zero
        for size in sizes {
            if [.top, .bottom].contains(alignment) {
                buttonsSize = CGSize(
                    width: max(size.width, buttonsSize.width),
                    height: buttonsSize.height + size.height + spacing
                )
            } else {
                buttonsSize = CGSize(
                    width: buttonsSize.width + size.width + spacing,
                    height: max(size.height, buttonsSize.height)
                )
            }
        }

        var wholeSize = CGSize.zero
        if [.top, .bottom].contains(alignment) {
            wholeSize = CGSize(
                width: max(buttonsSize.width, mainButtonSize.width),
                height: buttonsSize.height + mainButtonSize.height
            )
        } else {
            wholeSize = CGSize(
                width: buttonsSize.width + mainButtonSize.width,
                height: max(buttonsSize.height, mainButtonSize.height)
            )
        }

        menuButtonsSize.wrappedValue = buttonsSize
        wholeMenuSize.wrappedValue = wholeSize
    }

    fileprivate func roundToTwoDigits(_ size: CGSize) -> CGSize {
        CGSize(width: ceil(size.width*100)/100, height: ceil(size.height*100)/100)
    }
    
    fileprivate func calculateCoordsCircle() {
        guard sizes.count > 0, mainButtonSize != .zero else {
            return
        }

        let count = buttons.count
        var radius: Double = 60
        if let r = self.radius {
            radius = r
        } else if let buttonWidth = sizes.first?.width {
            radius = Double((mainButtonSize.width + buttonWidth) / 2 + spacing)
        }

        coords = (0..<count).map { i in
            let increment = (endAngle - startAngle) / Double(count - 1) * Double(i)
            let angle = layoutDirection == .clockwise ? startAngle + increment : startAngle - increment
            
            return CGPoint(x: radius*cos(angle), y: radius*sin(angle))
        }

        var finalFrame = CGRect(x: -mainButtonSize.width/2, y: -mainButtonSize.height/2, width: mainButtonSize.width, height: mainButtonSize.height)
        let buttonSize = sizes.first?.width ?? 0
        let buttonRadius = buttonSize / 2

        for coord in coords {
            finalFrame = finalFrame.union(CGRect(x: coord.x - buttonRadius, y: coord.y - buttonRadius, width: buttonSize, height: buttonSize))
        }

        wholeMenuSize.wrappedValue = finalFrame.size
        menuButtonsSize.wrappedValue = finalFrame.size
    }
}

public class DefaultFloatingButton { fileprivate init() {} }
public class StraightFloatingButton: DefaultFloatingButton {}
public class CircleFloatingButton: DefaultFloatingButton {}

public struct FloatingButtonGeneric<T: DefaultFloatingButton, MainView: View, ButtonView: View>: View {
    private var floatingButton: FloatingButton<MainView, ButtonView>
    
    fileprivate init(floatingButton: FloatingButton<MainView, ButtonView>) {
        self.floatingButton = floatingButton
    }
    
    fileprivate init() {
        fatalError("don't call this method")
    }
    
    public var body: some View {
        floatingButton
    }
}

public extension FloatingButton {
    
    func straight() -> FloatingButtonGeneric<StraightFloatingButton, MainView, ButtonView> {
        var copy = self
        copy.menuType = .straight
        return FloatingButtonGeneric(floatingButton: copy)
    }
    
    func circle() -> FloatingButtonGeneric<CircleFloatingButton, MainView, ButtonView> {
        var copy = self
        copy.menuType = .circle
        return FloatingButtonGeneric(floatingButton: copy)
    }
}

public extension FloatingButtonGeneric where T : DefaultFloatingButton {
    
    func spacing(_ spacing: CGFloat) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.spacing = spacing
        return copy
    }
    
    func initialScaling(_ initialScaling: CGFloat) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.initialScaling = initialScaling
        return copy
    }
    
    func initialOffset(_ initialOffset: CGPoint) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.initialOffset = initialOffset
        return copy
    }
    
    func initialOffset(x: CGFloat = 0, y: CGFloat = 0) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.initialOffset = CGPoint(x: x, y: y)
        return copy
    }
    
    func initialOpacity(_ initialOpacity: Double) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.initialOpacity = initialOpacity
        return copy
    }
    
    func animation(_ animation: Animation) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.animation = animation
        return copy
    }
    
    func delays(delayDelta: Double) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.delays = (0..<self.floatingButton.buttons.count).map { i in
            return delayDelta * Double(i)
        }
        return copy
    }
    
    func delays(_ delays: [Double]) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.delays = delays
        return copy
    }

    func mainZStackAlignment(_ alignment: SwiftUI.Alignment) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.mainZStackAlignment = alignment
        return copy
    }

    func inverseZIndex(_ inverse: Bool) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.inverseZIndex = inverse
        return copy
    }

    func wholeMenuSize(_ wholeMenuSize: Binding<CGSize>) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.wholeMenuSize = wholeMenuSize
        return copy
    }

    func menuButtonsSize(_ menuButtonsSize: Binding<CGSize>) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.menuButtonsSize = menuButtonsSize
        return copy
    }
}

public extension FloatingButtonGeneric where T: StraightFloatingButton {
    
    func direction(_ direction: Direction) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.direction = direction
        return copy
    }
    
    func alignment(_ alignment: Alignment) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.alignment = alignment
        return copy
    }
}

public extension FloatingButtonGeneric where T: CircleFloatingButton {
    
    func startAngle(_ startAngle: Double) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.startAngle = startAngle
        return copy
    }
    
    func endAngle(_ endAngle: Double) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.endAngle = endAngle
        return copy
    }
    
    func radius(_ radius: Double) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.radius = radius
        return copy
    }

    func layoutDirection(_ layoutDirection: LayoutDirection) -> FloatingButtonGeneric {
        var copy = self
        copy.floatingButton.layoutDirection = layoutDirection
        return copy
    }

}

struct SubmenuButton<ButtonView: View>: View {
    
    var buttonView: ButtonView
    var action: () -> Void = { }
    
    var body: some View {
        Button(action: { action() }) {
            buttonView
        }
        .buttonStyle(PlainButtonStyle())
    }
}

private struct MainButtonViewInternal<MainView: View>: View {
    
    @Binding public var isOpen: Bool
    fileprivate var mainButtonView: MainView
    
    public var body: some View {
        Button(action: { isOpen.toggle() }) {
            mainButtonView
        }
    }
}
