//
//  ChatHistoryFileManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/20.
//

import Foundation

class ChatHistoryFileManager {
    static let shared = ChatHistoryFileManager()
    
    private init() {}
    
    // Base directory for chat histories
    private var baseDirectory: URL {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documentsDirectory.appendingPathComponent("ChatHistories")
    }
    
    // Create a directory for a specific historyId if it doesn't already exist
    func createHistoryDirectory(for historyId: String) {
        let historyDirectory = baseDirectory.appendingPathComponent(historyId)
        
        if !FileManager.default.fileExists(atPath: historyDirectory.path) {
            do {
                try FileManager.default.createDirectory(at: historyDirectory, withIntermediateDirectories: true, attributes: nil)
                print("Created directory for historyId \(historyId)")
            } catch {
                print("Failed to create directory for historyId \(historyId): \(error)")
            }
        } else {
            print("Directory for historyId \(historyId) already exists.")
        }
    }

    // Generic method to copy a file from URL to the history directory
    func copyFile(from url: URL, for historyId: String) -> URL? {
        let historyDirectory = baseDirectory.appendingPathComponent(historyId)
        
        // Ensure the history directory exists
        createHistoryDirectory(for: historyId)
        
        // Extract the file name from the URL
        let fileName = url.lastPathComponent
        let destinationURL = historyDirectory.appendingPathComponent(fileName)
        
        // Check if the file already exists at the destination
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            print("File already exists at \(destinationURL), returning existing URL.")
            return destinationURL
        }
        
        // Check if source file exists before copying
        guard FileManager.default.fileExists(atPath: url.path) else {
            print("Source file does not exist at \(url.path)")
            return nil
        }
        
        do {
            try FileManager.default.copyItem(at: url, to: destinationURL)
            print("File copied to \(destinationURL)")
            return destinationURL
        } catch {
            print("Failed to copy file: \(error)")
        }
        return nil
    }
    
    // Validate if a file exists at the given URL
    func validateFileExists(at url: URL) -> Bool {
        return FileManager.default.fileExists(atPath: url.path)
    }
    
    // Get the correct file URL for a history file, checking if it exists
    func getValidFileURL(for fileName: String, historyId: String) -> URL? {
        let historyDirectory = baseDirectory.appendingPathComponent(historyId)
        let fileURL = historyDirectory.appendingPathComponent(fileName)
        
        if FileManager.default.fileExists(atPath: fileURL.path) {
            return fileURL
        }
        
        print("File not found at expected path: \(fileURL.path)")
        return nil
    }
    
    // Delete the history directory
    func deleteHistoryDirectory(for historyId: String) {
        let historyDirectory = baseDirectory.appendingPathComponent(historyId)
        
        do {
            try FileManager.default.removeItem(at: historyDirectory)
            print("Deleted history directory for historyId \(historyId)")
        } catch {
            print("Failed to delete history directory: \(error)")
        }
    }
    
    // Fetch all history directories
    func fetchAllHistoryDirectories() -> [String] {
        do {
            let directories = try FileManager.default.contentsOfDirectory(at: baseDirectory, includingPropertiesForKeys: nil)
            return directories.map { $0.lastPathComponent }
        } catch {
            print("Failed to fetch history directories: \(error)")
            return []
        }
    }
}
