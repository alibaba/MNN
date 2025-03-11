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
    
    init(modelPath: String, modelName: String = "") {
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
    
    func readIterations() -> Int {
        guard let data = try? Data(contentsOf: configFileURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return 20
        }
        
        return json["iterations"] as? Int ?? 20
    }
    
    func updateIterations(_ value: Int) {
        do {
            let data = try Data(contentsOf: configFileURL)
            var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            
            // 确保迭代次数为正整数
            let iterations = max(1, value)
            json["iterations"] = iterations
            
            // 写入文件
            let updatedData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
            try updatedData.write(to: configFileURL)
        } catch {
            print("Error updating iterations in config.json: \(error.localizedDescription)")
        }
    }
    
    func readSeed() -> Int {
        guard let data = try? Data(contentsOf: configFileURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return -1 // default random seed
        }
        
        return json["seed"] as? Int ?? -1
    }
    
    func updateSeed(_ value: Int) {
        do {
            let data = try Data(contentsOf: configFileURL)
            var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            
            json["seed"] = value
            
            let updatedData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
            try updatedData.write(to: configFileURL)
        } catch {
            print("Error updating seed in config.json: \(error.localizedDescription)")
        }
    }
}
