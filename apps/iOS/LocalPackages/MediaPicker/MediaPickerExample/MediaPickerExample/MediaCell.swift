//
//  MediaCell.swift
//  Example-iOS
//
//  Created by Alisa Mylnikova on 21.04.2023.
//

import SwiftUI
import ExyteMediaPicker
import AVFoundation
import _AVKit_SwiftUI

struct MediaCell: View {

    @StateObject var viewModel: MediaCellViewModel

    var body: some View {
        GeometryReader { g in
            VStack {
                if let url = viewModel.imageUrl {
                    AsyncImage(url: url) { phase in
                        if case let .success(image) = phase {
                            image
                                .resizable()
                                .scaledToFill()
                        }
                    }
                } else if let player = viewModel.player {
                    VideoPlayer(player: player).onTapGesture {
                        viewModel.togglePlay()
                    }
                } else {
                    ProgressView()
                }
            }
            .frame(width: g.size.width, height: g.size.height)
            .clipped()
        }
        .task {
            await viewModel.onStart()
        }
        .onDisappear {
            viewModel.onStop()
        }
    }
}

final class MediaCellViewModel: ObservableObject {

    let media: Media

    @Published var imageUrl: URL? = nil
    @Published var player: AVPlayer? = nil

    private var isPlaying = false

    init(media: Media) {
        self.media = media
    }

    func onStart() async {
        guard imageUrl == nil || player == nil else { return }

        let url = await media.getURL()
        guard let url = url else { return }

        DispatchQueue.main.async {
            switch self.media.type {
            case .image:
                self.imageUrl = url
            case .video:
                self.player = AVPlayer(url: url)
            }
        }
    }

    func togglePlay() {
        if isPlaying {
            player?.pause()
        } else {
            player?.play()
        }
        isPlaying = !isPlaying
    }

    func onStop() {
        imageUrl = nil
        player = nil
        isPlaying = false
    }
}
