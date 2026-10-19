//
//  View+fullscreenCover.swift
//  
//
//  Created by Alisa Mylnikova on 20.03.2023.
//

import SwiftUI

extension View {

    func transparentNonAnimatingFullScreenCover<Item, Content>(item: Binding<Item?>, @ViewBuilder content: @escaping () -> Content) -> some View where Item : Equatable, Item : Identifiable, Content : View {
        modifier(TransparentNonAnimatableFullScreenModifier(item: item, fullScreenContent: content))
    }
}

private struct TransparentNonAnimatableFullScreenModifier<Item, FullScreenContent>: ViewModifier where Item : Equatable, Item : Identifiable, FullScreenContent : View {

    @Binding var item: Item?
    let fullScreenContent: () -> (FullScreenContent)

    func body(content: Content) -> some View {
        content
            .onChange(of: item) { _ in
                UIView.setAnimationsEnabled(false)
            }
            .fullScreenCover(item: $item) { _ in
                ZStack {
                    fullScreenContent()
                }
                .background(FullScreenCoverBackgroundRemovalView())
                .onAppear {
                    if !UIView.areAnimationsEnabled {
                        UIView.setAnimationsEnabled(true)
                    }
                }
                .onDisappear {
                    if !UIView.areAnimationsEnabled {
                        UIView.setAnimationsEnabled(true)
                    }
                }
            }
    }

}

private struct FullScreenCoverBackgroundRemovalView: UIViewRepresentable {

    private class BackgroundRemovalView: UIView {
        override func didMoveToWindow() {
            super.didMoveToWindow()
            superview?.superview?.backgroundColor = .clear
        }
    }

    func makeUIView(context: Context) -> UIView {
        return BackgroundRemovalView()
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

}
