//
//  EqualizerIndicatorView.swift
//  ActivityIndicatorView
//
//  Created by Daniil Manin on 10/7/20.
//  Copyright © 2020 Exyte. All rights reserved.
//

import SwiftUI

struct EqualizerIndicatorView: View {

    let count: Int

    var body: some View {
        GeometryReader { geometry in
            ForEach(0..<count, id: \.self) { index in
                EqualizerIndicatorItemView(index: index, count: count, size: geometry.size)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}

struct EqualizerIndicatorItemView: View {

    let index: Int
    let count: Int
    let size: CGSize

    @State private var scale: CGFloat = 0

    var body: some View {
        let itemSize = size.width / CGFloat(count) / 2

        let animation = Animation.easeOut.delay(0.2)
            .repeatForever(autoreverses: true)
            .delay(Double(index) / Double(count) / 2)

        return RoundedRectangle(cornerRadius: 3)
            .frame(width: itemSize, height: size.height)
            .scaleEffect(x: 1, y: scale, anchor: .center)
            .onAppear {
                scale = 1
                withAnimation(animation) {
                    scale = 0.4
                }
            }
            .offset(x: 2 * itemSize * CGFloat(index) - size.width / 2 + itemSize / 2)
    }
}
