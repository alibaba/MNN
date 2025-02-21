//
//  ModelConfigManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/17.
//

import Foundation

class ModelConfigManager {
    private let modelPath: String
    private let configFileName = "config.json"
    
    init(modelPath: String) {
        self.modelPath = modelPath
    }
    
    private var configFileURL: URL {
        URL(fileURLWithPath: modelPath).appendingPathComponent(configFileName)
    }
    
    func readUseMmap() -> Bool {
        guard let data = try? Data(contentsOf: configFileURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        
        return json["use_mmap"] as? Bool ?? true
    }
    
    func updateUseMmap(_ value: Bool) {
        do {
            let data = try Data(contentsOf: configFileURL)
            var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            
            // Update or add use_mmap field
            json["use_mmap"] = value
            
            // Write back to file
            let updatedData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
            try updatedData.write(to: configFileURL)
        } catch {
            print("Error updating config.json: \(error.localizedDescription)")
        }
    }
}
