import Foundation
import SwiftUI

/// How well a model is expected to run here, judged from its weight size
/// against physical RAM. Without the increased-memory-limit entitlement an app
/// gets roughly half of RAM before jetsam, and runtime buffers plus the KV
/// cache come on top of the weights.
enum ModelDeviceFit {
    case recommended
    case heavy
    case tooLarge

    static func evaluate(modelBytes: Int64,
                         physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) -> ModelDeviceFit {
        let ratio = Double(modelBytes) / Double(max(physicalMemory, 1))
        if ratio <= 0.25 { return .recommended }
        if ratio <= 0.40 { return .heavy }
        return .tooLarge
    }

    var title: String {
        switch self {
        case .recommended: AppText.text("fit.recommended")
        case .heavy: AppText.text("fit.heavy")
        case .tooLarge: AppText.text("fit.tooLarge")
        }
    }

    var systemImage: String {
        switch self {
        case .recommended: "checkmark.circle"
        case .heavy: "tortoise"
        case .tooLarge: "exclamationmark.triangle"
        }
    }

    var color: Color {
        switch self {
        case .recommended: .green
        case .heavy: .orange
        case .tooLarge: .red
        }
    }
}

enum ByteText {
    static func file(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: max(bytes, 0), countStyle: .file)
    }

    static var deviceMemory: String {
        ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory)
    }
}

enum StorageSpace {
    /// Space the system is willing to free up for user-initiated content.
    static func availableBytes() -> Int64? {
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let values = try? documents.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        else { return nil }
        return values.volumeAvailableCapacityForImportantUsage
    }
}

extension ModelInfo {
    /// Total download size in bytes, when the catalog or a size probe knows it.
    var expectedBytes: Int64? {
        if let fileSize = file_size, fileSize > 0 { return Int64(fileSize) }
        if let cachedSize, cachedSize > 0 { return cachedSize }
        if let sizeGB = size_gb, sizeGB > 0 { return Int64(sizeGB * 1024 * 1024 * 1024) }
        return nil
    }

    /// Nil for bundled models and when the size is unknown.
    var deviceFit: ModelDeviceFit? {
        guard vendor != "Local", let bytes = expectedBytes else { return nil }
        return ModelDeviceFit.evaluate(modelBytes: bytes)
    }
}
