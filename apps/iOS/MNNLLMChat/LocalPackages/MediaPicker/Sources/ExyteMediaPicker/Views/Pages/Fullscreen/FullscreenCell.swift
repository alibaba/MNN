//
//  Created by Alex.M on 09.06.2022.
//

import Foundation
import SwiftUI
import AVKit

struct FullscreenCell: View {
    
    @Environment(\.mediaPickerTheme) private var theme

    @StateObject var viewModel: FullscreenCellViewModel
    @ObservedObject var keyboardHeightHelper = KeyboardHeightHelper.shared

    var body: some View {
        GeometryReader { g in
            Group {
                if let image = viewModel.image {
                    let useFill = g.size.width / g.size.height > image.size.width / image.size.height
                    ZoomableScrollView {
                        imageView(image: image, useFill: useFill)
                    }
                } else if let player = viewModel.player {
                    let useFill = g.size.width / g.size.height > viewModel.videoSize.width / viewModel.videoSize.height
                    ZoomableScrollView {
                        videoView(player: player, useFill: useFill)
                    }
                } else {
                    ProgressView()
                        .tint(.white)
                }
            }
            .allowsHitTesting(!keyboardHeightHelper.keyboardDisplayed)
            .position(x: g.frame(in: .local).midX, y: g.frame(in: .local).midY)
        }
        .task {
            await viewModel.onStart()
        }
        .onDisappear {
            viewModel.onStop()
        }
    }

    @ViewBuilder
    func imageView(image: UIImage, useFill: Bool) -> some View {
        Image(uiImage: image)
            .resizable()
            .aspectRatio(contentMode: useFill ? .fill : .fit)
    }

    func videoView(player: AVPlayer, useFill: Bool) -> some View {
        PlayerView(player: player, bgColor: theme.main.fullscreenPhotoBackground, useFill: useFill)
            .disabled(true)
            .overlay {
                ZStack {
                    Color.clear
                    if !viewModel.isPlaying {
                        Image(systemName: "play.fill")
                            .resizable()
                            .frame(width: 50, height: 50)
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    viewModel.togglePlay()
                }
            }
    }
}
