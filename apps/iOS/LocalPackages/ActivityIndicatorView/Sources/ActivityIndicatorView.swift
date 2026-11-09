//
//  ActivityIndicatorView.swift
//  ActivityIndicatorView
//
//  Created by Alisa Mylnikova on 20/03/2020.
//  Copyright © 2020 Exyte. All rights reserved.
//

import SwiftUI

public struct ActivityIndicatorView: View {

    public enum IndicatorType {
        case `default`(count: Int = 8)
        case arcs(count: Int = 3, lineWidth: CGFloat = 2)
        case rotatingDots(count: Int = 5)
        case flickeringDots(count: Int = 8)
        case scalingDots(count: Int = 3, inset: Int = 2)
        case opacityDots(count: Int = 3, inset: Int = 4)
        case equalizer(count: Int = 5)
        case growingArc(Color = .black, lineWidth: CGFloat = 4)
        case growingCircle
        case gradient(_ colors: [Color], CGLineCap = .butt, lineWidth: CGFloat = 4)
    }

    @Binding var isVisible: Bool
    var type: IndicatorType

    public init(isVisible: Binding<Bool>, type: IndicatorType) {
        _isVisible = isVisible
        self.type = type
    }

    public var body: some View {
        if isVisible {
            indicator
        } else {
            EmptyView()
        }
    }
    
    // MARK: - Private
    
    private var indicator: some View {
        ZStack {
            switch type {
            case .default(let count):
                DefaultIndicatorView(count: count)
            case .arcs(let count, let lineWidth):
                ArcsIndicatorView(count: count, lineWidth: lineWidth)
            case .rotatingDots(let count):
                RotatingDotsIndicatorView(count: count)
            case .flickeringDots(let count):
                FlickeringDotsIndicatorView(count: count)
            case .scalingDots(let count, let inset):
                ScalingDotsIndicatorView(count: count, inset: inset)
            case .opacityDots(let count, let inset):
                OpacityDotsIndicatorView(count: count, inset: inset)
            case .equalizer(let count):
                EqualizerIndicatorView(count: count)
            case .growingArc(let color, let lineWidth):
                GrowingArcIndicatorView(color: color, lineWidth: lineWidth)
            case .growingCircle:
                GrowingCircleIndicatorView()
            case .gradient(let colors, let lineCap, let lineWidth):
                GradientIndicatorView(colors: colors, lineCap: lineCap, lineWidth: lineWidth)
            }
        }
    }
}
