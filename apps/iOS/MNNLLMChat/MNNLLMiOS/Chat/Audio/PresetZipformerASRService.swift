//
//  PresetZipformerASRService.swift
//  MNNLLMiOS
//

import AVFoundation
import CryptoKit
import ExyteChat
import Foundation

struct PresetASRRecognition {
    let transcript: String
    let recording: Recording
    let audioDurationSeconds: Double
    let modelInitializationSeconds: Double
    let inferenceSeconds: Double

    var realTimeFactor: Double {
        guard audioDurationSeconds > 0 else { return 0 }
        return inferenceSeconds / audioDurationSeconds
    }

    var realTimeMultiple: Double {
        guard inferenceSeconds > 0 else { return 0 }
        return audioDurationSeconds / inferenceSeconds
    }

    var displayText: String {
        let transcriptLine = String(
            format: NSLocalizedString("ASR 识别：%@", comment: "Preset ASR transcript"),
            transcript
        )
        let speedLine = String(
            format: NSLocalizedString(
                "ASR %.3fs · 音频 %.3fs · RTF %.3f · %.1f× 实时",
                comment: "Preset ASR performance summary"
            ),
            inferenceSeconds,
            audioDurationSeconds,
            realTimeFactor,
            realTimeMultiple
        )
        return "\(transcriptLine)\n\(speedLine)"
    }
}

enum PresetASRError: LocalizedError {
    case missingAsset(String)
    case invalidAsset(String)
    case invalidAudio(String)
    case initializationFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingAsset(let path):
            return "缺少 ASR 资源：\(path)"
        case .invalidAsset(let path):
            return "ASR 资源校验失败：\(path)"
        case .invalidAudio(let reason):
            return "预设音频无法读取：\(reason)"
        case .initializationFailed(let reason):
            return "X-ASR 初始化失败：\(reason)"
        }
    }
}

final class PresetZipformerASRService: @unchecked Sendable {
    private struct AudioSamples {
        let pcm: Data
        let sampleRate: Int32
        let duration: Double
        let waveform: [CGFloat]
    }

    private let workerQueue = DispatchQueue(label: "com.alibaba.mnnchat.preset-asr", qos: .userInitiated)
    private let stateLock = NSLock()
    private var wrapper: PresetZipformerASRWrapper?
    private var assetsValidated = false
    private var reportedInitialization = false

    func recognize(audioURL: URL, expectedAudioSHA256: String) async throws -> PresetASRRecognition {
        try await withCheckedThrowingContinuation { continuation in
            workerQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: PresetASRError.initializationFailed("ASR service released"))
                    return
                }
                do {
                    let result = try self.recognizeOnWorker(
                        audioURL: audioURL,
                        expectedAudioSHA256: expectedAudioSHA256
                    )
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    func cancel() {
        stateLock.lock()
        let activeWrapper = wrapper
        stateLock.unlock()
        activeWrapper?.cancel()
    }

    func shutdown() {
        stateLock.lock()
        let activeWrapper = wrapper
        wrapper = nil
        assetsValidated = false
        stateLock.unlock()
        activeWrapper?.shutdown()
    }

    private func recognizeOnWorker(audioURL: URL, expectedAudioSHA256: String) throws -> PresetASRRecognition {
        let audioHash = try sha256(of: audioURL)
        guard audioHash == expectedAudioSHA256 else {
            throw PresetASRError.invalidAsset(audioURL.lastPathComponent)
        }
        let audio = try loadMonoFloatPCM(from: audioURL)
        let modelDirectory = try resolveAndValidateModelDirectory()
        let activeWrapper = try getOrCreateWrapper(modelDirectory: modelDirectory)

        var inferenceSeconds: TimeInterval = 0
        let rawTranscript = try activeWrapper.recognizeFloatPCM(
            audio.pcm,
            sampleRate: audio.sampleRate,
            inferenceSeconds: &inferenceSeconds
        )
        let transcript = normalizeXASRTranscript(rawTranscript)
        let initializationSeconds: Double
        stateLock.lock()
        if reportedInitialization {
            initializationSeconds = 0
        } else {
            reportedInitialization = true
            initializationSeconds = activeWrapper.initializationSeconds
        }
        stateLock.unlock()

        NSLog(
            "[PRESET_ASR] event=recognized runtime=sherpa-mnn model=x-asr-zipformer2 " +
                "precision=mnn-weight-int8-block64 backend=cpu " +
                "threads=2 input_rate=%d input_samples=%d audio_sha256=%@ transcript=%@ init_ms=%.1f " +
                "inference_ms=%.1f audio_ms=%.1f rtf=%.4f",
            audio.sampleRate,
            audio.pcm.count / MemoryLayout<Float>.size,
            audioHash,
            transcript,
            initializationSeconds * 1000,
            inferenceSeconds * 1000,
            audio.duration * 1000,
            inferenceSeconds / audio.duration
        )

        return PresetASRRecognition(
            transcript: transcript,
            recording: Recording(duration: audio.duration, waveformSamples: audio.waveform, url: audioURL),
            audioDurationSeconds: audio.duration,
            modelInitializationSeconds: initializationSeconds,
            inferenceSeconds: inferenceSeconds
        )
    }

    private func getOrCreateWrapper(modelDirectory: URL) throws -> PresetZipformerASRWrapper {
        stateLock.lock()
        if let wrapper {
            stateLock.unlock()
            return wrapper
        }
        stateLock.unlock()

        let candidate = PresetZipformerASRWrapper(modelDirectory: modelDirectory.path)
        guard candidate.isReady else {
            throw PresetASRError.initializationFailed(candidate.initializationError ?? "unknown error")
        }
        stateLock.lock()
        wrapper = candidate
        stateLock.unlock()
        return candidate
    }

    private func resolveAndValidateModelDirectory() throws -> URL {
        guard let resourcePath = Bundle.main.resourceURL else {
            throw PresetASRError.missingAsset("Bundle resource root")
        }
        let directory = resourcePath.appendingPathComponent("LocalModel/xasr-mnn-int8", isDirectory: true)

        stateLock.lock()
        let alreadyValidated = assetsValidated
        stateLock.unlock()
        if alreadyValidated {
            return directory
        }

        let manifest: [(String, String)] = [
            ("encoder-160ms.mnn", "5570b93b89969665c0428cc3c159e41bc71ded7a297e75b1fe09c6c6dce68c82"),
            ("decoder-160ms.mnn", "83f4bb5fc1b28130af1823c7b20937ce2b789923ccc2beb2fbc09b81f7de2359"),
            ("joiner-160ms.mnn", "3f71a9408890199ed3fb71b2a29bb3aada57f81281a6cfbdf753d859a1a1fa33"),
            ("tokens.txt", "b818a60878b9aae978cbb8ad594acbd403d76d1af2e31ef4197c84e2dbdba27c"),
        ]
        for (file, expectedHash) in manifest {
            let url = directory.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw PresetASRError.missingAsset(url.path)
            }
            guard try sha256(of: url) == expectedHash else {
                throw PresetASRError.invalidAsset(url.path)
            }
            NSLog("[PRESET_ASR] event=asset_verified file=%@ sha256=%@", url.path, expectedHash)
        }
        stateLock.lock()
        assetsValidated = true
        stateLock.unlock()
        return directory
    }

    private func normalizeXASRTranscript(_ text: String) -> String {
        let cjk = "\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff"
        let punctuation = "，。！？；：、（）《》〈〉【】「」『』“”‘’"
        let patterns = [
            "(?<=[\(cjk)])\\s+(?=[\(cjk)])",
            "(?<=[\(cjk)])\\s+(?=[\(punctuation)])",
            "(?<=[\(punctuation)])\\s+(?=[\(cjk)])",
            "(?<=[\(punctuation)])\\s+(?=[\(punctuation)])",
            "\\s+(?=[,.!?;:%)\\]}])",
        ]
        return patterns.reduce(text) { current, pattern in
            current.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func loadMonoFloatPCM(from url: URL) throws -> AudioSamples {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard format.channelCount == 1 else {
            throw PresetASRError.invalidAudio("expected mono, got \(format.channelCount) channels")
        }
        guard file.length > 0, file.length <= AVAudioFramePosition(UInt32.max) else {
            throw PresetASRError.invalidAudio("invalid frame count \(file.length)")
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw PresetASRError.invalidAudio("failed to allocate PCM buffer")
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0 else {
            throw PresetASRError.invalidAudio("Float32 PCM is unavailable")
        }
        let count = Int(buffer.frameLength)
        let pcm = Data(bytes: channel, count: count * MemoryLayout<Float>.size)
        let duration = Double(count) / format.sampleRate
        return AudioSamples(
            pcm: pcm,
            sampleRate: Int32(format.sampleRate.rounded()),
            duration: duration,
            waveform: waveformSamples(channel, count: count, bins: 36)
        )
    }

    private func waveformSamples(_ samples: UnsafePointer<Float>, count: Int, bins: Int) -> [CGFloat] {
        guard count > 0, bins > 0 else { return [] }
        let stride = max(1, count / bins)
        return (0..<bins).map { bin in
            let start = min(count, bin * stride)
            let end = min(count, start + stride)
            guard start < end else { return 0 }
            var peak: Float = 0
            for index in start..<end {
                peak = max(peak, abs(samples[index]))
            }
            return CGFloat(min(1, peak))
        }
    }

    private func sha256(of url: URL) throws -> String {
        guard let stream = InputStream(url: url) else {
            throw PresetASRError.missingAsset(url.path)
        }
        stream.open()
        defer { stream.close() }
        var digest = SHA256()
        var bytes = [UInt8](repeating: 0, count: 1024 * 1024)
        while stream.hasBytesAvailable {
            let count = stream.read(&bytes, maxLength: bytes.count)
            if count < 0 {
                throw stream.streamError ?? PresetASRError.invalidAsset(url.path)
            }
            if count == 0 {
                break
            }
            digest.update(data: Data(bytes[0..<count]))
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
