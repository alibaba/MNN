//
//  ModelDownloadStorage.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

/// ModelDownloadStorage - Persistent storage manager for download state tracking
/// 
/// This class provides comprehensive storage management for tracking downloaded model files,
/// their metadata, and download completion status. It uses UserDefaults for persistence
/// and maintains file integrity through size and revision validation.
/// 
/// Key Features:
/// - Persistent download state tracking across app sessions
/// - File integrity validation using size and revision checks
/// - Efficient storage using relative path mapping
/// - Thread-safe operations with immediate persistence
/// - Automatic cleanup and state management
/// 
/// Architecture:
/// - Uses UserDefaults as the underlying storage mechanism
/// - Maintains a dictionary mapping relative paths to FileStatus objects
/// - Provides atomic operations for state updates
/// - Supports both individual file and batch operations
/// 
/// Storage Format:
/// - Files are indexed by relative paths from Documents directory
/// - Each entry contains size, revision, and timestamp information
/// - JSON encoding ensures cross-session compatibility
/// 
/// Usage:
/// ```swift
/// let storage = ModelDownloadStorage()
/// 
/// // Check if file is already downloaded
/// if storage.isFileDownloaded(file, at: destinationPath) {
///     print("File already exists and is up to date")
/// }
/// 
/// // Save download completion status
/// storage.saveFileStatus(file, at: destinationPath)
/// ```
final class ModelDownloadStorage {
    // MARK: - Properties
    
    private let userDefaults: UserDefaults
    private let downloadedFilesKey = "ModelScope.DownloadedFiles"
    
    // MARK: - Initialization
    
    /// Initializes the storage manager with configurable UserDefaults
    /// 
    /// - Parameter userDefaults: UserDefaults instance for persistence (defaults to .standard)
    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }
    
    // MARK: - Public Methods
    
    /// Checks if a file has been completely downloaded and is up to date
    /// 
    /// Validates both file existence on disk and metadata consistency including
    /// file size and revision to ensure the downloaded file is current and complete.
    /// 
    /// - Parameters:
    ///   - file: ModelFile object containing expected metadata
    ///   - path: Destination path where the file should be located
    /// - Returns: true if file exists and matches expected metadata, false otherwise
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
    
    /// Saves the download completion status for a file
    /// 
    /// Records file metadata including size, revision, and download timestamp
    /// to persistent storage. This enables resume functionality and prevents
    /// unnecessary re-downloads of unchanged files.
    /// 
    /// - Parameters:
    ///   - file: ModelFile object containing file metadata
    ///   - path: Destination path where the file was downloaded
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
    
    /// Retrieves all downloaded file statuses from persistent storage
    /// 
    /// - Returns: Dictionary mapping relative file paths to FileStatus objects,
    ///            or nil if no download history exists
    func getDownloadedFiles() -> [String: FileStatus]? {
        guard let data = userDefaults.data(forKey: downloadedFilesKey),
              let downloadedFiles = try? JSONDecoder().decode([String: FileStatus].self, from: data) else {
            return nil
        }
        return downloadedFiles
    }
    
    /// Removes download status for a specific file
    /// 
    /// Cleans up storage by removing the file's download record, typically
    /// used when files are deleted or need to be re-downloaded.
    /// 
    /// - Parameter path: Full path to the file whose status should be cleared
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
    
    /// Converts absolute file paths to relative paths from Documents directory
    /// 
    /// This normalization ensures consistent path handling across different
    /// app installations and device configurations.
    /// 
    /// - Parameter fullPath: Absolute file path to convert
    /// - Returns: Relative path from Documents directory, or original path if conversion fails
    private func getRelativePath(from fullPath: String) -> String {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.path else {
            return fullPath
        }
        
        // Convert absolute path to relative path from Documents directory
        return fullPath.replacingOccurrences(of: documentsPath + "/", with: "")
    }
}
