//
//  BatchFileTestService.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/10/13.
//

import Foundation

/// Service class for batch file operations
class BatchFileTestService {
    /// Generates an output file with the given results and configuration
    /// - Parameters:
    ///   - results: Array of test results
    ///   - configuration: Output configuration
    /// - Returns: URL of the generated file
    func generateOutputFile(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileName = "batch_test_results_\(Int(Date().timeIntervalSince1970)).\(configuration.outputFormat.rawValue)"
        let fileURL = documentsPath.appendingPathComponent(fileName)

        let content: String

        switch configuration.outputFormat {
        case .txt:
            content = generateTextOutput(results: results, configuration: configuration)
        case .json:
            content = try generateJSONOutput(results: results, configuration: configuration)
        case .jsonl:
            content = try generateJSONLOutput(results: results, configuration: configuration)
        }

        try content.write(to: fileURL, atomically: true, encoding: .utf8)
        return fileURL
    }

    /// Generates text format output
    private func generateTextOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) -> String {
        var lines: [String] = []

        for (index, result) in results.enumerated() {
            lines.append("=== Test \(index + 1) ===")
            
            lines.append("Prompt: \(result.item.prompt)")

            lines.append("Response: \(result.response)")

            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    /// Generates JSON format output
    private func generateJSONOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> String {
        let jsonResults = results.map { result -> [String: Any] in
            var dict: [String: Any] = [
                "response": result.response,
            ]

            if configuration.includePrompt {
                dict["prompt"] = result.item.prompt
            }

            if configuration.includeTimestamp {
                dict["timestamp"] = ISO8601DateFormatter().string(from: result.timestamp)
            }

            return dict
        }

        let jsonData = try JSONSerialization.data(withJSONObject: jsonResults, options: .prettyPrinted)
        return String(data: jsonData, encoding: .utf8) ?? ""
    }

    /// Generates JSONL format output
    private func generateJSONLOutput(results: [BatchTestResult], configuration: BatchTestConfiguration) throws -> String {
        var lines: [String] = []

        for result in results {
            var dict: [String: Any] = [
                "response": result.response,
            ]

            if configuration.includePrompt {
                dict["prompt"] = result.item.prompt
            }

            if configuration.includeTimestamp {
                dict["timestamp"] = ISO8601DateFormatter().string(from: result.timestamp)
            }

            let jsonData = try JSONSerialization.data(withJSONObject: dict)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                lines.append(jsonString)
            }
        }

        return lines.joined(separator: "\n")
    }
    
    /// Processes media references (images and audios) in JSON data generically
    /// - Parameters:
    ///   - json: JSON dictionary containing media references
    ///   - prompt: Original prompt string
    ///   - baseURL: Base URL for resolving relative paths
    /// - Returns: processedPrompt containing processed prompt
    func processMediaReferences(json: [String: Any], prompt: String, baseURL: URL?) -> String {
        var processedPrompt = prompt
        
        // Process all image references
        let imageKeys = json.keys.filter { $0.hasPrefix("image") }
        for imageKey in imageKeys.sorted() {
            guard let imagePath = json[imageKey] as? String else { continue }
            
            // Extract the number from the key (e.g., "image1" -> "1", "image10" -> "10")
            let numberString = String(imageKey.dropFirst(5)) // Remove "image" prefix
            guard !numberString.isEmpty else { continue }
            
            // Process the image reference if baseURL is available
            if let baseURL = baseURL {
                let fullImagePath = baseURL.appendingPathComponent(imagePath).path
                let placeholder = "<image \(numberString)>"
                let replacement = "<img>\(fullImagePath)</img>"
                processedPrompt = processedPrompt.replacingOccurrences(of: placeholder, with: replacement)
            }
        }
        
        // Process all audio references
        let audioKeys = json.keys.filter { $0.hasPrefix("audio") }
        for audioKey in audioKeys.sorted() {
            guard let audioPath = json[audioKey] as? String else { continue }
            
            // Extract the number from the key (e.g., "audio1" -> "1", "audio10" -> "10")
            let numberString = String(audioKey.dropFirst(5)) // Remove "audio" prefix
            guard !numberString.isEmpty else { continue }
            
            // Process the audio reference if baseURL is available
            if let baseURL = baseURL {
                let fullAudioPath = baseURL.appendingPathComponent(audioPath).path
                let placeholder = "<audio \(numberString)>"
                let replacement = "<audio>\(fullAudioPath)</audio>"
                processedPrompt = processedPrompt.replacingOccurrences(of: placeholder, with: replacement)
            }
        }
        
        return processedPrompt
    }
}
