//
//  DownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

public class DownloadStorage {
    private let userDefaults: UserDefaults
    private let downloadedFilesKey = "ModelScope.DownloadedFiles"
    
    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }
    
    private func getRelativePath(from fullPath: String) -> String {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.path else {
            return fullPath
        }
        
        return fullPath.replacingOccurrences(of: documentsPath + "/", with: "")
    }
    
    func isFileDownloaded(_ file: ModelFile, at path: String) -> Bool {
        let filePath = (path as NSString)
            .appendingPathComponent(file.name.sanitizedPath)
        let relativePath = getRelativePath(from: filePath)
        
        guard FileManager.default.fileExists(atPath: filePath),
              let downloadedFiles = getDownloadedFiles(),
              let fileStatus = downloadedFiles[relativePath] else {
            return false
        }
        
        return fileStatus.size == file.size &&
               fileStatus.revision == file.revision
    }
    
    func saveFileStatus(_ file: ModelFile, at path: String) {
        let filePath = (path as NSString)
            .appendingPathComponent(file.name.sanitizedPath)
        let relativePath = getRelativePath(from: filePath)
        
        let fileStatus = FileStatus(
            path: relativePath,
            size: file.size,
            revision: file.revision,
            lastModified: Date()
        )
        
        var downloadedFiles = getDownloadedFiles() ?? [:]
        downloadedFiles[relativePath] = fileStatus
        
        if let encoded = try? JSONEncoder().encode(downloadedFiles) {
            userDefaults.set(encoded, forKey: downloadedFilesKey)
            userDefaults.synchronize()
        }
        print("Saved file status for: \(relativePath)")
        print("Current downloaded files: \(downloadedFiles)")
    }
    
    func getDownloadedFiles() -> [String: FileStatus]? {
        guard let data = userDefaults.data(forKey: downloadedFilesKey),
              let downloadedFiles = try? JSONDecoder().decode([String: FileStatus].self, from: data) else {
            print("Failed to get downloaded files from UserDefaults")
            return nil
        }
        print("Retrieved downloaded files: \(downloadedFiles)")
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
}
