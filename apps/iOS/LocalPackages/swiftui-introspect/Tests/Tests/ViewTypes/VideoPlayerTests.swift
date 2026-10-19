#if canImport(AVKit)
import AVKit
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, tvOS 14, macOS 11, *)
@MainActor
final class VideoPlayerTests: XCTestCase {
    #if canImport(UIKit)
    typealias PlatformVideoPlayer = AVPlayerViewController
    #elseif canImport(AppKit)
    typealias PlatformVideoPlayer = AVPlayerView
    #endif

    func testVideoPlayer() throws {
        guard #available(iOS 14, tvOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        let videoURL0 = URL(string: "https://bit.ly/swswift#1")!
        let videoURL1 = URL(string: "https://bit.ly/swswift#2")!
        let videoURL2 = URL(string: "https://bit.ly/swswift#3")!

        XCTAssertViewIntrospection(of: PlatformVideoPlayer.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            VStack {
                VideoPlayer(player: AVPlayer(url: videoURL0))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.videoPlayer, on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy0)
                    #elseif os(macOS)
                    .introspect(.videoPlayer, on: .macOS(.v11, .v12, .v13, .v14), customize: spy0)
                    #endif

                VideoPlayer(player: AVPlayer(url: videoURL1))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.videoPlayer, on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy1)
                    #elseif os(macOS)
                    .introspect(.videoPlayer, on: .macOS(.v11, .v12, .v13, .v14), customize: spy1)
                    #endif

                VideoPlayer(player: AVPlayer(url: videoURL2))
                    #if os(iOS) || os(tvOS) || os(visionOS)
                    .introspect(.videoPlayer, on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .visionOS(.v1, .v2), customize: spy2)
                    #elseif os(macOS)
                    .introspect(.videoPlayer, on: .macOS(.v11, .v12, .v13, .v14), customize: spy2)
                    #endif
            }
        } extraAssertions: {
            XCTAssertEqual(($0[safe: 0]?.player?.currentItem?.asset as? AVURLAsset)?.url, videoURL0)
            XCTAssertEqual(($0[safe: 1]?.player?.currentItem?.asset as? AVURLAsset)?.url, videoURL1)
            XCTAssertEqual(($0[safe: 2]?.player?.currentItem?.asset as? AVURLAsset)?.url, videoURL2)
        }
    }
}
#endif
