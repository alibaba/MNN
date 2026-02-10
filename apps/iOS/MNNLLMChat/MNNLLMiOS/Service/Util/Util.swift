//
//  Util.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/7.
//

import ExyteChat
import ExyteMediaPicker

import Foundation
import AVFoundation

func convertToWavFormat(inputUrl: URL, outputUrl: URL) throws {

    let inputFile = try AVAudioFile(forReading: inputUrl)
    
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
    let outputFile = try AVAudioFile(forWriting: outputUrl, settings: format.settings)
    
    let buffer = AVAudioPCMBuffer(pcmFormat: inputFile.processingFormat, frameCapacity: 1024)!
    
    if inputFile.processingFormat.sampleRate != format.sampleRate || inputFile.processingFormat.channelCount != format.channelCount {
        throw NSError(domain: "AudioConversionError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Input format does not match output format."])
    }
    
    while true {
        try inputFile.read(into: buffer)
        if buffer.frameLength == 0 { break }
        try outputFile.write(from: buffer)
    }
}

func convertACCToWAV(accFileUrl: URL) async -> URL? {
    let directoryUrl = accFileUrl.deletingLastPathComponent()
    let wavFileName = accFileUrl.deletingPathExtension().lastPathComponent + ".wav"
    let wavFileUrl = directoryUrl.appendingPathComponent(wavFileName)
    
    // 如果目标文件已存在，先删除
    try? FileManager.default.removeItem(at: wavFileUrl)
    
    let asset = AVAsset(url: accFileUrl)
    
    // 使用 AVAssetExportPresetPassthrough 预设
    guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
        print("Failed to create AVAssetExportSession.")
        return nil
    }
    
    // 设置输出格式为 caf，后续再转换为 wav
    exportSession.outputFileType = .caf
    exportSession.outputURL = wavFileUrl
    
    // 使用 async/await 替代 DispatchGroup
    await withCheckedContinuation { continuation in
        exportSession.exportAsynchronously {
            switch exportSession.status {
            case .completed:
                print("CAF file successfully created at: \(wavFileUrl.path)")
            case .failed:
                if let error = exportSession.error {
                    print("Error during export: \(error.localizedDescription)")
                }
            case .cancelled:
                print("Export cancelled.")
            default:
                print("Export status: \(exportSession.status.rawValue)")
            }
            continuation.resume()
        }
    }
    
    // 检查文件是否存在
    if FileManager.default.fileExists(atPath: wavFileUrl.path) {
        do {
            // 使用 AVAudioFile 将 CAF 转换为 WAV
            let inputFile = try AVAudioFile(forReading: wavFileUrl)
            let format = AVAudioFormat(standardFormatWithSampleRate: inputFile.processingFormat.sampleRate,
                                     channels: inputFile.processingFormat.channelCount)!
            let outputFile = try AVAudioFile(forWriting: wavFileUrl, settings: format.settings)
            
            let buffer = AVAudioPCMBuffer(pcmFormat: inputFile.processingFormat,
                                        frameCapacity: AVAudioFrameCount(inputFile.length))!
            
            try inputFile.read(into: buffer)
            try outputFile.write(from: buffer)
            
            return wavFileUrl
        } catch {
            print("Error converting to WAV: \(error)")
            return nil
        }
    } else {
        print("CAF file does not exist at: \(wavFileUrl.path)")
        return nil
    }
}


extension DraftMessage {
    func makeLLMChatImages() async -> [LLMChatImage] {
        await medias
            .filter { $0.type == .image }
            .asyncMap { (media : Media) -> (Media, URL?, URL?) in
                (media, await media.getThumbnailURL(), await media.getURL())
            }
            .compactMap { media, thumb, full in
                guard let thumb, let full else { return nil }
                return LLMChatImage(id: media.id.uuidString, thumbnail: thumb, full: full)
            }
    }

    func makeLLMChatVideos() async -> [LLMChatVideo] {
        await medias
            .filter { $0.type == .video }
            .asyncMap { (media : Media) -> (Media, URL?, URL?) in
                (media, await media.getThumbnailURL(), await media.getURL())
            }
            .compactMap { media, thumb, full in
                guard let thumb, let full else { return nil }
                return LLMChatVideo(id: media.id.uuidString, thumbnail: thumb, full: full)
            }
    }

    func toLLMChatMessage(id: String, user: LLMChatUser, status: Message.Status = .read) async -> LLMChatMessage {
        LLMChatMessage(
            uid: id,
            sender: user,
            createdAt: createdAt,
            status: user.isCurrentUser ? status : nil,
            text: text,
            images: await makeLLMChatImages(),
            videos: await makeLLMChatVideos(),
            recording: recording,
            replyMessage: replyMessage
        )
    }
}

/// MARK: Util

class DateFormatting {
    static var agoFormatter = RelativeDateTimeFormatter()
}

extension Date {
    // 1 hour ago, 2 days ago...
    func formatAgo() -> String {
        let result = DateFormatting.agoFormatter.localizedString(for: self, relativeTo: Date())
        if result.contains("second") {
            return "Just now"
        }
        
        if result.contains("秒钟") {
            return "刚刚"
        }
        
        return result
    }
}


extension Sequence {
    func asyncMap<T>(
        _ transform: (Element) async throws -> T
    ) async rethrows -> [T] {
        var values = [T]()

        for element in self {
            try await values.append(transform(element))
        }

        return values
    }
}


