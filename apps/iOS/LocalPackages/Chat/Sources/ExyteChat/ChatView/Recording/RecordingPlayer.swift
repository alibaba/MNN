//
//  RecordingPlayer.swift
//
//
//  Created by Alexandra Afonasova on 21.06.2022.
//

import Combine
import AVFoundation

final class RecordingPlayer: NSObject, ObservableObject {

    @Published var playing = false
    @Published var duration: Double = 0.0
    @Published var secondsLeft: Double = 0.0
    @Published var progress: Double = 0.0

    private let audioSession = AVAudioSession()

    var didPlayTillEnd = PassthroughSubject<Void, Never>()

    private var recording: Recording?

    private var player: AVPlayer?
    private var timeObserver: Any?

    override init() {
        super.init()
        do {
            try audioSession.setCategory(.playback, mode: .default, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            print("[RecordingPlayer] Error configuring audio session: \(error)")
        }
    }

    func play(_ recording: Recording) {
        guard let url = recording.url else { 
            return 
        }
        
        do {
            let audioSession = AVAudioSession.sharedInstance()
            
            try audioSession.setCategory(.playback, mode: .default, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
        } catch {
            print("[RecordingPlayer] Failed to activate audio session: \(error)")
        }
        
        self.recording = recording
        setupPlayer(for: url, trackDuration: recording.duration)
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
            self?.play()
        }
    }

    func pause() {
        player?.pause()
        playing = false
    }

    func togglePlay(_ recording: Recording) {
        if self.recording?.url != recording.url {
            self.recording = recording
            if let url = recording.url {
                setupPlayer(for: url, trackDuration: recording.duration)
            }
        }
        if playing { pause() }
        else { play() }
    }

    func seek(with recording: Recording, to progress: Double) {
        let goalTime = recording.duration * progress
        if self.recording == nil {
            self.recording = recording
            if let url = recording.url {
                setupPlayer(for: url, trackDuration: recording.duration)
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.player?.seek(to: CMTime(seconds: goalTime, preferredTimescale: 10))
                if self?.playing == nil || self?.playing == false  {
                    self?.play()
                }
            }
            return
        }
        self.player?.seek(to: CMTime(seconds: goalTime, preferredTimescale: 10))
        if !self.playing {
            self.play()
        }
    }

    func seek(to progress: Double) {
        let goalTime = duration * progress
        player?.seek(to: CMTime(seconds: goalTime, preferredTimescale: 10))
        if !playing { play() }
    }

    func reset() {
        if playing {
            pause()
        }
        recording = nil
        progress = 0
    }

    private func play() {
        do {
            try audioSession.setActive(true)
        } catch {
            print("[RecordingPlayer] Error activating audio session: \(error)")
        }
        
        player?.play()
        playing = true
        NotificationCenter.default.post(name: .chatAudioIsPlaying, object: self)
    }

    private func setupPlayer(for url: URL, trackDuration: Double) {
        duration = trackDuration
        progress = 0.0
        secondsLeft = trackDuration
        NotificationCenter.default.removeObserver(self)
        timeObserver = nil
        player?.replaceCurrentItem(with: nil)

        let playerItem = AVPlayerItem(url: url)
        player = AVPlayer(playerItem: playerItem)
        
        // Add observer for player item status changes
        playerItem.addObserver(self, forKeyPath: "status", options: [.new, .initial], context: nil)
        
        NotificationCenter.default.addObserver(forName: .chatAudioIsPlaying, object: nil, queue: .main) { notification in
            if let sender = notification.object as? RecordingPlayer {
                if sender.recording?.url == self.recording?.url {
                    return
                }
                self.pause()
            }
        }

        NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: playerItem,
            queue: nil
        ) { [weak self] _ in
            self?.playing = false
            self?.player?.seek(to: .zero)
            self?.didPlayTillEnd.send()
        }

        timeObserver = player?.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.2, preferredTimescale: 10),
            queue: DispatchQueue.main
        ) { [weak self] time in
            guard let item = self?.player?.currentItem, !item.duration.seconds.isNaN else { return }
            self?.duration = item.duration.seconds
            self?.progress = time.seconds / item.duration.seconds
            self?.secondsLeft = (item.duration - time).seconds.rounded()
        }
    }
    
    // MARK: - KVO Observer
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
        if keyPath == "status" {
            if let playerItem = object as? AVPlayerItem {
                switch playerItem.status {
                case .readyToPlay:
                    let audioTracks = playerItem.asset.tracks(withMediaType: .audio)
                    
                    self.player?.volume = 1.0
                    
                    for (index, _) in audioTracks.enumerated() {
                        if index < playerItem.tracks.count {
                            let playerItemTrack = playerItem.tracks[index]
                            playerItemTrack.isEnabled = true
                        }
                    }
                    
                case .failed:
                    if let error = playerItem.error {
                        print("[RecordingPlayer] Player item failed with error: \(error.localizedDescription)")
                        if let underlyingError = (error as NSError).userInfo[NSUnderlyingErrorKey] as? NSError {
                            print("[RecordingPlayer] Underlying error: \(underlyingError.localizedDescription)")
                        }
                    }
                    
                case .unknown:
                    break
                    
                @unknown default:
                    break
                }
            }
        }
    }
    
    deinit {
        // Remove observers
        NotificationCenter.default.removeObserver(self)
        if let timeObserver = timeObserver {
            player?.removeTimeObserver(timeObserver)
        }
        // Remove KVO observer
        player?.currentItem?.removeObserver(self, forKeyPath: "status")
    }

}
