//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Foundation
import os.log

public final class ModelDownloadLogger {
    
    static var isEnabled: Bool = false
    private static let logger = Logger(subsystem: Bundle.main.bundleIdentifier ?? "ModelScope", category: "Download")
    
    static func debug(_ message: String) {
        guard isEnabled else { return }
        logger.debug("📥 \(message)")
    }
    
    static func info(_ message: String) {
        guard isEnabled else { return }
        logger.info("ℹ️ \(message)")
    }
    
    static func error(_ message: String) {
        guard isEnabled else { return }
        logger.error("❌ \(message)")
    }
} 
