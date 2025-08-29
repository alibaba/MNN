//
//  FileOperationManager.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import UIKit

/// FileOperationManager is a singleton utility class that handles various file operations
/// including image processing, audio processing, directory size calculation, and file cleanup.
final class FileOperationManager {
    
    /// Shared singleton instance
    static let shared = FileOperationManager()
    
    /// Private initializer to enforce singleton pattern
    private init() {}
    
    // MARK: - Image Processing
    
    /// Processes image files by copying to temporary directory and performing HEIC conversion if needed
    /// 
    /// - Parameters:
    ///   - url: The original image URL
    ///   - fileName: The desired file name for the processed image
    /// - Returns: The processed image URL, or nil if processing fails
    /// 
    /// Usage:
    /// ```swift
    /// let imageURL = URL(fileURLWithPath: "/path/to/image.heic")
    /// if let processedURL = FileOperationManager.shared.processImageFile(from: imageURL, fileName: "converted.jpg") {
    ///     // Use the processed image URL
    /// }
    /// ```
    func processImageFile(from url: URL, fileName: String) -> URL? {
        let isInTempDirectory = url.path.contains("/tmp/")
        
        if !isInTempDirectory {
            guard let fileUrl = AssetExtractor.copyFileToTmpDirectory(from: url, fileName: fileName) else {
                return nil
            }
            return convertHEICImage(from: fileUrl)
        } else {
            return convertHEICImage(from: url)
        }
    }
    
    /// Converts HEIC images to JPG format using AssetExtractor utility
    /// 
    /// - Parameter url: The HEIC image URL to convert
    /// - Returns: The converted JPG image URL, or original URL if not HEIC format
    private func convertHEICImage(from url: URL) -> URL? {
        var fileUrl = url
        if fileUrl.isHEICImage() {
            if let convertedUrl = AssetExtractor.convertHEICToJPG(heicUrl: fileUrl) {
                fileUrl = convertedUrl
            }
        }
        return fileUrl
    }
    
    
    // MARK: - Directory Size Calculation
    
    /// Formats byte count into human-readable string using ByteCountFormatter
    /// 
    /// - Parameter bytes: The number of bytes to format
    /// - Returns: Formatted string (e.g., "1.5 GB")
    /// 
    /// Usage:
    /// ```swift
    /// let size: Int64 = 1073741824 // 1 GB
    /// let formatted = FileOperationManager.shared.formatBytes(size)
    /// print(formatted) // "1.0 GB"
    /// ```
    func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }
    
    /// Calculates the size of a local directory and returns a formatted string
    /// 
    /// - Parameter path: The directory path to calculate size for
    /// - Returns: Formatted size string or "Unknown" if calculation fails
    /// 
    /// Usage:
    /// ```swift
    /// let directoryPath = "/path/to/directory"
    /// let sizeString = FileOperationManager.shared.formatLocalDirectorySize(at: directoryPath)
    /// print("Directory size: \(sizeString)")
    /// ```
    func formatLocalDirectorySize(at path: String) -> String {
        guard FileManager.default.fileExists(atPath: path) else { return "Unknown" }
        
        do {
            let totalSize = try calculateDirectorySize(at: path)
            return formatBytes(totalSize)
        } catch {
            return "Unknown"
        }
    }
    
    /// Calculates the total size of a directory by traversing all files recursively
    /// Uses actual disk allocated size when available, falls back to logical file size
    /// 
    /// - Parameter path: The directory path to calculate size for
    /// - Returns: Total directory size in bytes
    /// - Throws: FileSystem errors during directory traversal
    /// 
    /// Usage:
    /// ```swift
    /// do {
    ///     let size = try FileOperationManager.shared.calculateDirectorySize(at: "/path/to/directory")
    ///     print("Directory size: \(size) bytes")
    /// } catch {
    ///     print("Failed to calculate directory size: \(error)")
    /// }
    /// ```
    func calculateDirectorySize(at path: String) throws -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        // print("Calculating directory size for path: \(path)")
        
        let directoryURL = URL(fileURLWithPath: path)
        
        guard fileManager.fileExists(atPath: path) else {
            // print("Path does not exist: \(path)")
            return 0
        }
        
        let resourceKeys: [URLResourceKey] = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey, .nameKey]
        let enumerator = fileManager.enumerator(
            at: directoryURL,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { (url, error) -> Bool in
                print("Error accessing \(url): \(error)")
                return true
            }
        )
        
        guard let fileEnumerator = enumerator else {
            throw NSError(domain: "FileEnumerationError", code: -1, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create file enumerator"])
        }
        
        var fileCount = 0
        for case let fileURL as URL in fileEnumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: Set(resourceKeys))
                
                guard let isRegularFile = resourceValues.isRegularFile, isRegularFile else { continue }
                
                let fileName = resourceValues.name ?? "Unknown"
                fileCount += 1
                
                // Use actual disk allocated size, fallback to logical size if not available
                if let actualSize = resourceValues.totalFileAllocatedSize {
                    totalSize += Int64(actualSize)
                    
                    if fileCount <= 10 {
                        let actualSizeGB = Double(actualSize) / (1024 * 1024 * 1024)
                        let logicalSizeGB = Double(resourceValues.fileSize ?? 0) / (1024 * 1024 * 1024)
                        // print("File \(fileCount): \(fileName) - Logical: \(String(format: "%.3f", logicalSizeGB)) GB, Actual: \(String(format: "%.3f", actualSizeGB)) GB")
                    }
                } else if let logicalSize = resourceValues.fileSize {
                    totalSize += Int64(logicalSize)
                    
                    if fileCount <= 10 {
                        let logicalSizeGB = Double(logicalSize) / (1024 * 1024 * 1024)
                        // print("File \(fileCount): \(fileName) - Size: \(String(format: "%.3f", logicalSizeGB)) GB (fallback)")
                    }
                }
            } catch {
                print("Error getting resource values for \(fileURL): \(error)")
                continue
            }
        }
        
        let totalSizeGB = Double(totalSize) / (1024 * 1024 * 1024)
        print("Total files: \(fileCount), Total actual disk usage: \(String(format: "%.2f", totalSizeGB)) GB")
        
        return totalSize
    }
    
    // MARK: - Directory Cleaning
    
    /// Cleans temporary directories based on memory mapping usage
    /// Cleans system temporary directory and optionally model temporary directories
    ///
    /// 
    /// Usage:
    /// ```swift
    /// // Clean temporary directories
    /// FileOperationManager.shared.cleanTempDirectories()
    /// ```
    func cleanTempDirectories() {
        let fileManager = FileManager.default
        let tmpDirectoryURL = fileManager.temporaryDirectory
        
        cleanFolder(at: tmpDirectoryURL)
    }
    
    /// Cleans the temporary folder for a specific model
    /// 
    /// - Parameter modelPath: The path to the model directory
    /// 
    /// Usage:
    /// ```swift
    /// let modelPath = "/path/to/model"
    /// FileOperationManager.shared.cleanModelTempFolder(modelPath: modelPath)
    /// ```
    func cleanModelTempFolder(modelPath: String) {
        let tmpFolderURL = URL(fileURLWithPath: modelPath).appendingPathComponent("temp")
        cleanFolder(at: tmpFolderURL)
    }
    
    /// Recursively cleans all files in the specified folder
    /// Preserves files containing "networkdownload" in their path
    /// 
    /// - Parameter folderURL: The folder URL to clean
    private func cleanFolder(at folderURL: URL) {
        let fileManager = FileManager.default
        do {
            let files = try fileManager.contentsOfDirectory(at: folderURL, includingPropertiesForKeys: nil)
            for file in files {
                if !file.absoluteString.lowercased().contains("networkdownload") {
                    do {
                        try fileManager.removeItem(at: file)
                        print("Deleted file: \(file.path)")
                    } catch {
                        print("Error deleting file: \(file.path), \(error.localizedDescription)")
                    }
                }
            }
        } catch {
            print("Error accessing directory: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Diffusion Image Generation
    
    /// Generates a unique temporary file path for Diffusion model image output
    /// Creates a unique JPG filename in the system temporary directory
    /// 
    /// - Returns: A unique temporary image file URL
    /// 
    /// Usage:
    /// ```swift
    /// let tempImageURL = FileOperationManager.shared.generateTempImagePath()
    /// // Use tempImageURL for image generation output
    /// ```
    func generateTempImagePath() -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let imageName = UUID().uuidString + ".jpg"
        return tempDir.appendingPathComponent(imageName)
    }
}
