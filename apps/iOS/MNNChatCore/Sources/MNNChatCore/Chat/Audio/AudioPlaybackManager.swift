//
//  AudioPlaybackManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/12/5.
//  Created for Omni audio output support
//

import AVFoundation
import Foundation

/// Manages audio playback for Omni model audio output
/// Receives PCM float data and plays it using AVAudioEngine
class AudioPlaybackManager {
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioFormat: AVAudioFormat?
    private var isPlaying: Bool = false
    private let sampleRate: Double = 24000.0 // Omni models use 24kHz
    
    var onPlaybackComplete: (() -> Void)?
    
    init() {
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default, options: [])
            try audioSession.setActive(true)
            print("[AudioPlayback] Audio session configured: category=playback, mode=default")
        } catch {
            print("[AudioPlayback] Failed to setup audio session: \(error)")
        }
    }
    
    /// Start audio playback engine
    func start() {
        guard audioEngine == nil else {
            print("[AudioPlayback] Audio engine already started, skipping")
            return
        }
        
        print("[AudioPlayback] Starting audio engine...")
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        
        guard let engine = audioEngine, let node = playerNode else {
            print("[AudioPlayback] Failed to create audio engine or player node")
            return
        }
        
        // Create audio format: mono, 24kHz, float32
        audioFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: sampleRate,
                                   channels: 1,
                                   interleaved: false)
        
        guard let format = audioFormat else {
            print("[AudioPlayback] Failed to create audio format")
            return
        }
        
        print("[AudioPlayback] Audio format: sampleRate=\(sampleRate), channels=1, format=Float32")
        
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: format)
        
        do {
            try engine.start()
            node.play()
            isPlaying = true
            print("[AudioPlayback] Audio engine started successfully, isRunning=\(engine.isRunning)")
        } catch {
            print("[AudioPlayback] Failed to start audio engine: \(error)")
        }
    }
    
    /// Play a chunk of PCM float data
    /// - Parameters:
    ///   - data: PCM float array
    ///   - isLastChunk: Whether this is the last chunk
    func playChunk(data: [Float], isLastChunk: Bool) {
        guard let node = playerNode, let format = audioFormat, let engine = audioEngine else {
            print("[AudioPlayback] Cannot play chunk: node=\(playerNode != nil), format=\(audioFormat != nil), engine=\(audioEngine != nil)")
            return
        }
        
        // Ensure engine is running
        if !engine.isRunning {
            print("[AudioPlayback] Audio engine is not running, attempting to restart...")
            do {
                try engine.start()
                node.play()
                isPlaying = true
                print("[AudioPlayback] Audio engine restarted successfully")
            } catch {
                print("[AudioPlayback] Failed to restart audio engine: \(error)")
                return
            }
        }
        
        // Ensure node is playing
        if !isPlaying {
            print("[AudioPlayback] Node not playing, starting playback...")
            node.play()
            isPlaying = true
        }
        
        print("[AudioPlayback] Received audio chunk: size=\(data.count), isLastChunk=\(isLastChunk), engineRunning=\(engine.isRunning)")
        
        // Data should already be validated and filtered in ViewModel callback
        // But log statistics for debugging
        let maxVal = data.max() ?? 0
        let minVal = data.min() ?? 0
        let avgVal = data.reduce(0, +) / Float(data.count)
        let nonZeroCount = data.filter { abs($0) > 0.0001 }.count
        print("[AudioPlayback] Audio data stats: min=\(minVal), max=\(maxVal), avg=\(avgVal), nonZero=\(nonZeroCount)/\(data.count)")
        
        // Convert Float array to AVAudioPCMBuffer
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(data.count)) else {
            print("[AudioPlayback] Failed to create PCM buffer for \(data.count) samples")
            return
        }
        
        buffer.frameLength = AVAudioFrameCount(data.count)
        
        // Copy float data to buffer
        guard let channelData = buffer.floatChannelData else {
            print("[AudioPlayback] Failed to get channel data from buffer")
            return
        }
        
        memcpy(channelData[0], data, data.count * MemoryLayout<Float>.size)
        
        // Schedule buffer for playback
        node.scheduleBuffer(buffer) { [weak self] in
            print("[AudioPlayback] Buffer playback completed, isLastChunk=\(isLastChunk)")
            if isLastChunk {
                DispatchQueue.main.async {
                    print("[AudioPlayback] Last chunk completed, stopping playback")
                    self?.stop()
                    self?.onPlaybackComplete?()
                }
            }
        }
        
        print("[AudioPlayback] Scheduled buffer for playback: \(data.count) frames, buffer.frameLength=\(buffer.frameLength)")
    }
    
    /// Stop audio playback
    func stop() {
        guard let node = playerNode, let engine = audioEngine else {
            print("[AudioPlayback] Stop called but engine/node not initialized")
            return
        }
        
        print("[AudioPlayback] Stopping audio playback...")
        node.stop()
        engine.stop()
        engine.reset()
        
        audioEngine = nil
        playerNode = nil
        audioFormat = nil
        isPlaying = false
        
        print("[AudioPlayback] Audio playback stopped and cleaned up")
    }
    
    /// Reset and restart audio engine
    func reset() {
        stop()
        start()
    }
    
    deinit {
        stop()
    }
}

