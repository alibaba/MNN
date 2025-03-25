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
    
    private let defaultTfsZ: Double = 1.0
    private let defaultTypical: Double = 1.0
    private let defaultPenalty: Double = 0.0
    private let defaultNGram: Int = 8
    private let defaultNGramFactor: Double = 1.0
    
    init(modelPath: String, modelName: String = "") {
        self.modelPath = modelPath
    }
    
    private var configFileURL: URL {
        URL(fileURLWithPath: modelPath).appendingPathComponent(configFileName)
    }
    
    private func readValue<T>(_ key: String, defaultValue: T) -> T {
        guard let data = try? Data(contentsOf: configFileURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return defaultValue
        }
        return json[key] as? T ?? defaultValue
    }

    private func updateValue<T>(_ key: String, value: T) {
        do {
            let data = try Data(contentsOf: configFileURL)
            var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            json[key] = value
            let updatedData = try JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)
            try updatedData.write(to: configFileURL)
        } catch {
            print("Error updating \(key) in config.json: \(error.localizedDescription)")
        }
    }
    
    // MARK: - UseMmap
    func readUseMmap() -> Bool {
        return readValue("use_mmap", defaultValue: false)
    }
    
    func updateUseMmap(_ value: Bool) {
        updateValue("use_mmap", value: value)
    }
    
    // MARK: - Iterations
    func readIterations() -> Int {
        return readValue("iterations", defaultValue: 20)
    }
    
    func updateIterations(_ value: Int) {
        updateValue("iterations", value: max(1, value))
    }
    
    // MARK: - Seed
    func readSeed() -> Int {
        return readValue("seed", defaultValue: -1)
    }
    
    func updateSeed(_ value: Int) {
        updateValue("seed", value: value)
    }
    
    // MARK: - Temperature
    func readTemperature() -> Double {
        return readValue("temperature", defaultValue: 1.0)
    }
    
    func updateTemperature(_ value: Double) {
        updateValue("temperature", value: max(0.0, min(value, 2.0)))
    }
    
    // MARK: - TopK
    func readTopK() -> Int {
        return readValue("topK", defaultValue: 40)
    }
    
    func updateTopK(_ value: Int) {
        updateValue("topK", value: max(1, value))
    }
    
    // MARK: - TopP
    func readTopP() -> Double {
        return readValue("topP", defaultValue: 0.9)
    }
    
    func updateTopP(_ value: Double) {
        updateValue("topP", value: max(0.0, min(value, 1.0)))
    }
    
    // MARK: - MinP
    func readMinP() -> Double {
        return readValue("minP", defaultValue: 0.1)
    }
    
    func updateMinP(_ value: Double) {
        updateValue("minP", value: max(0.0, min(value, 1.0)))
    }
    
    // MARK: - TFS-Z
    func readTfsZ() -> Double {
        return readValue("tfsZ", defaultValue: defaultTfsZ)
    }
    
    func updateTfsZ(_ value: Double) {
        updateValue("tfsZ", value: value)
    }
    
    // MARK: - Typical
    func readTypical() -> Double {
        return readValue("typical", defaultValue: defaultTypical)
    }
    
    func updateTypical(_ value: Double) {
        updateValue("typical", value: value)
    }
    
    // MARK: - Penalty
    func readPenalty() -> Double {
        return readValue("penalty", defaultValue: defaultPenalty)
    }
    
    func updatePenalty(_ value: Double) {
        updateValue("penalty", value: value)
    }
    
    // MARK: - N-gram
    func readNGram() -> Int {
        return readValue("nGram", defaultValue: defaultNGram)
    }
    
    func updateNGram(_ value: Int) {
        updateValue("nGram", value: value)
    }
    
    // MARK: - N-gram Factor
    func readNGramFactor() -> Double {
        return readValue("nGramFactor", defaultValue: defaultNGramFactor)
    }
    
    func updateNGramFactor(_ value: Double) {
        updateValue("nGramFactor", value: value)
    }
    
    // MARK: - Read all config string
    func readConfigAsJSONString() -> String? {
        do {
            let data = try Data(contentsOf: configFileURL)
            if let jsonString = String(data: data, encoding: .utf8) {
                print("Config JSON String: \(jsonString)") // debug
                return jsonString
            }
    
        } catch {
            print("Error reading config file: \(error.localizedDescription)")
        }
        return nil
    }
    
    // MARK: - SamplerType
    func readSamplerType() -> SamplerType {
        let typeString = readValue("sampler_type", defaultValue: "temperature")
        return SamplerType(rawValue: typeString) ?? .temperature
    }
    
    func updateSamplerType(_ value: SamplerType) {
        updateValue("sampler_type", value: value.rawValue)
    }
    
    // MARK: - MixedSamplers
    func readMixedSamplers() -> [String] {
        return readValue("mixed_samplers", defaultValue: ["topK", "tfs", "typical", "topP", "minP", "temperature"])
    }
    
    func updateMixedSamplers(_ value: [String]) {
        updateValue("mixed_samplers", value: value)
    }
    
    // MARK: - PenaltySampler
    func readPenaltySampler() -> PenaltySamplerType {
        let typeString = readValue("penalty_sampler", defaultValue: "greedy")
        return PenaltySamplerType(rawValue: typeString) ?? .greedy
    }
    
    func updatePenaltySampler(_ value: PenaltySamplerType) {
        updateValue("penalty_sampler", value: value.rawValue)
    }
}

public enum SamplerType: String, CaseIterable {
    case greedy = "greedy"
    case temperature = "temperature"
    case topK = "topK"
    case topP = "topP"
    case minP = "minP"
    case tfs = "tfs"
    case typical = "typical"
    case penalty = "penalty"
    case mixed = "mixed"
    
    var displayName: String {
        switch self {
        case .greedy: return "Greedy"
        case .temperature: return "Temperature"
        case .topK: return "Top-K"
        case .topP: return "Top-P"
        case .minP: return "Min-P"
        case .tfs: return "TFS"
        case .typical: return "Typical"
        case .penalty: return "Penalty"
        case .mixed: return "Mixed"
        }
    }
}

enum PenaltySamplerType: String, CaseIterable {
    case greedy = "greedy"
    case temperature = "temperature"
    
    var displayName: String {
        switch self {
        case .greedy: return "Greedy"
        case .temperature: return "Temperature"
        }
    }
}
