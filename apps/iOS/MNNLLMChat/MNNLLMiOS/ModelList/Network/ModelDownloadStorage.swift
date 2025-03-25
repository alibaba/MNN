//
//  ModelDownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

final class ModelDownloadStorage {
    // MARK: - Properties
    
    private let userDefaults: UserDefaults
    private let downloadedFilesKey = "ModelScope.DownloadedFiles"
    
    // MARK: - Initialization
    
    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }
    
    // MARK: - Public Methods
    
    func isFileDownloaded(_ file: ModelFile, at path: String) -> Bool {
        let filePath = (path as NSString).appendingPathComponent(file.name.sanitizedPath)
        let relativePath = getRelativePath(from: filePath)
        
        guard FileManager.default.fileExists(atPath: filePath),
              let downloadedFiles = getDownloadedFiles(),
              let fileStatus = downloadedFiles[relativePath] else {
            return false
        }
        
        return fileStatus.size == file.size && fileStatus.revision == file.revision
    }
    
    func saveFileStatus(_ file: ModelFile, at path: String) {
        let filePath = (path as NSString).appendingPathComponent(file.name.sanitizedPath)
        let relativePath = getRelativePath(from: filePath)
        
        let fileStatus = FileStatus(
            path: relativePath,
            size: file.size,
            revision: file.revision,
            lastModified: Date()
        )
        
        var downloadedFiles = getDownloadedFiles() ?? [:]
        downloadedFiles[relativePath] = fileStatus
        
        // Save to UserDefaults and ensure immediate write
        if let encoded = try? JSONEncoder().encode(downloadedFiles) {
            userDefaults.set(encoded, forKey: downloadedFilesKey)
            userDefaults.synchronize()
        }
    }
    
    func getDownloadedFiles() -> [String: FileStatus]? {
        guard let data = userDefaults.data(forKey: downloadedFilesKey),
              let downloadedFiles = try? JSONDecoder().decode([String: FileStatus].self, from: data) else {
            return nil
        }
        return downloadedFiles
    }
    
    func clearFileStatus(at path: String) {
        let relativePath = getRelativePath(from: path)
        var downloadedFiles = getDownloadedFiles() ?? [:]
        downloadedFiles.removeValue(forKey: relativePath)
        
        if let encoded = try? JSONEncoder().encode(downloadedFiles) {
            userDefaults.set(encoded, forKey: downloadedFilesKey)
            userDefaults.synchronize()
        }
    }
    
    // MARK: - Private Methods
    
    private func getRelativePath(from fullPath: String) -> String {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.path else {
            return fullPath
        }
        
        // Convert absolute path to relative path from Documents directory
        return fullPath.replacingOccurrences(of: documentsPath + "/", with: "")
    }
}
