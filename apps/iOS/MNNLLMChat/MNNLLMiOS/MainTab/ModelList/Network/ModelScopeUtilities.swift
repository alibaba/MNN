//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

// MARK: - String Extensions
extension String {
    var sanitizedPath: String {
        self.removingPercentEncoding?
            .replacingOccurrences(of: "%20", with: " ")
            .replacingOccurrences(of: "%25", with: "%") ?? self
    }
}

// MARK: - FileManager Extensions
extension FileManager {
    func createDirectoryIfNeeded(at path: String) throws {
        guard !fileExists(atPath: path) else { return }
        try createDirectory(
            atPath: path,
            withIntermediateDirectories: true,
            attributes: nil
        )
    }
}


