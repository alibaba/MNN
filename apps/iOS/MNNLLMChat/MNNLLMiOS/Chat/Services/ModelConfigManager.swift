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

    private let defaultBackendType = "cpu"
    private let defaultPrecision = "low"
    private let defaultThreadNum = 4
    private let defaultMaxNewTokens = 128
    private let maximumMaxNewTokens = 4096
    private let defaultTfsZ: Double = 1.0
    private let defaultTypical: Double = 1.0
    private let defaultPenalty: Double = 0.0
    private let defaultNGram: Int = 8
    private let defaultNGramFactor: Double = 1.0
    private let defaultMultimodalPromptHint: String = "You are provided a set of visual and/or audio inputs. Please analyze them carefully before responding."
    private let defaultUseMultimodalPromptAPI: Bool = false

    init(modelPath: String, modelName _: String = "") {
        self.modelPath = modelPath
    }

    private var configFileURL: URL {
        URL(fileURLWithPath: modelPath).appendingPathComponent(configFileName)
    }

    // MARK: - Config paths

    /// Writable custom config stored under Documents/MNNConfigs/{modelName}/custom_config.json
    private var customConfigURL: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        let configDir = documentsPath
            .appendingPathComponent("MNNConfigs")
            .appendingPathComponent(modelName)

        // Ensure directory exists
        try? FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        return configDir.appendingPathComponent("custom_config.json")
    }

    /// Reads a value with custom settings taking precedence even when an older
    /// app version stored the setting under a legacy key.
    private func readValue<T>(_ key: String, legacyKeys: [String] = [], defaultValue: T) -> T {
        let candidateKeys = [key] + legacyKeys

        if FileManager.default.fileExists(atPath: customConfigURL.path),
           let customData = try? Data(contentsOf: customConfigURL),
           let customJson = try? JSONSerialization.jsonObject(with: customData) as? [String: Any] {
            for candidateKey in candidateKeys {
                if let value = customJson[candidateKey] as? T {
                    return value
                }
            }
        }

        if let data = try? Data(contentsOf: configFileURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for candidateKey in candidateKeys {
                if let value = json[candidateKey] as? T {
                    return value
                }
            }
        }

        return defaultValue
    }

    /// Updates only the writable custom_config.json with the provided key/value.
    private func updateValue<T>(_ key: String, value: T) {
        do {
            var customConfig: [String: Any] = [:]

            if FileManager.default.fileExists(atPath: customConfigURL.path),
               let data = try? Data(contentsOf: customConfigURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                customConfig = json
            }

            customConfig[key] = value

            let updatedData = try JSONSerialization.data(withJSONObject: customConfig, options: .prettyPrinted)
            try updatedData.write(to: customConfigURL)
        } catch {
            print("Error updating \(key) in custom config: \(error.localizedDescription)")
        }
    }

    // MARK: - Backend / Precision / Threads

    /// Maximum allowed threads based on current device CPU cores.
    var maxThreads: Int {
        max(1, ProcessInfo.processInfo.activeProcessorCount)
    }

    func readBackendType() -> String {
        return readValue("backend_type", defaultValue: defaultBackendType)
    }

    func updateBackendType(_ value: String) {
        updateValue("backend_type", value: value)
    }

    func readPrecision() -> String {
        return readValue("precision", defaultValue: defaultPrecision)
    }

    func updatePrecision(_ value: String) {
        updateValue("precision", value: value)
    }

    func readThreadNum() -> Int {
        return readValue("thread_num", defaultValue: defaultThreadNum)
    }

    func updateThreadNum(_ value: Int) {
        let clamped = max(1, min(maxThreads, value))
        updateValue("thread_num", value: clamped)
    }

    // MARK: - Maximum Output Tokens

    func readMaxNewTokens() -> Int {
        let value: Int = readValue("max_new_tokens", defaultValue: defaultMaxNewTokens)
        return max(1, min(maximumMaxNewTokens, value))
    }

    func updateMaxNewTokens(_ value: Int) {
        let clamped = max(1, min(maximumMaxNewTokens, value))
        updateValue("max_new_tokens", value: clamped)
    }

    func readPower() -> String {
        return readValue("power", defaultValue: "normal")
    }

    func readMemory() -> String {
        return readValue("memory", defaultValue: "low")
    }

    func readDynamicOption() -> Int {
        return readValue("dynamic_option", defaultValue: 0)
    }

    // MARK: - UseMmap

    func readUseMmap() -> Bool {
        return readValue("use_mmap", defaultValue: false)
    }

    func updateUseMmap(_ value: Bool) {
        updateValue("use_mmap", value: value)
    }

    // MARK: - Video Frames

    func readVideoMaxFrames() -> Int {
        return readValue("video_max_frames", defaultValue: 8)
    }

    func saveVideoMaxFrames(_ value: Int) {
        let clamped: Int = max(1, min(32, value))
        updateValue("video_max_frames", value: clamped)
    }

    // MARK: - Multimodal Prompt Hint

    func readDefaultMultimodalPrompt() -> String {
        return readValue("default_multimodal_prompt", defaultValue: defaultMultimodalPromptHint)
    }

    func saveDefaultMultimodalPrompt(_ value: String) {
        let trimmed: String = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalValue: String = trimmed.isEmpty ? defaultMultimodalPromptHint : trimmed
        updateValue("default_multimodal_prompt", value: finalValue)
    }

    func readUseMultimodalPromptAPI() -> Bool {
        return readValue("use_multimodal_prompt_api", defaultValue: defaultUseMultimodalPromptAPI)
    }

    func saveUseMultimodalPromptAPI(_ value: Bool) {
        updateValue("use_multimodal_prompt_api", value: value)
    }

    // MARK: - Audio Output (Omni)

    private let defaultEnableAudioOutput: Bool = false
    private let defaultTalkerSpeaker: String = "default"

    func readEnableAudioOutput() -> Bool {
        let value = readValue("enable_audio_output", defaultValue: defaultEnableAudioOutput)
        print("[AudioConfig] Read enable_audio_output: \(value)")
        return value
    }

    func saveEnableAudioOutput(_ value: Bool) {
        print("[AudioConfig] Save enable_audio_output: \(value)")
        updateValue("enable_audio_output", value: value)
    }

    func readTalkerSpeaker() -> String {
        let value = readValue("talker_speaker", defaultValue: defaultTalkerSpeaker)
        print("[AudioConfig] Read talker_speaker: \(value)")
        return value
    }

    func saveTalkerSpeaker(_ value: String) {
        print("[AudioConfig] Save talker_speaker: \(value)")
        updateValue("talker_speaker", value: value)
    }

    // MARK: - Iterations

    func readIterations() -> Int {
        return readValue("iterations", defaultValue: 10
        )
    }

    func updateIterations(_ value: Int) {
        updateValue("iterations", value: max(1, value))
    }

    // MARK: - Seed

    func readSeed() -> Int {
        return readValue("seed", defaultValue: 42)
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
        return readValue("top_k", legacyKeys: ["topK"], defaultValue: 40)
    }

    func updateTopK(_ value: Int) {
        updateValue("top_k", value: max(1, value))
    }

    // MARK: - TopP

    func readTopP() -> Double {
        return readValue("top_p", legacyKeys: ["topP"], defaultValue: 0.9)
    }

    func updateTopP(_ value: Double) {
        updateValue("top_p", value: max(0.0, min(value, 1.0)))
    }

    // MARK: - MinP

    func readMinP() -> Double {
        return readValue("min_p", legacyKeys: ["minP"], defaultValue: 0.1)
    }

    func updateMinP(_ value: Double) {
        updateValue("min_p", value: max(0.0, min(value, 1.0)))
    }

    // MARK: - TFS-Z

    func readTfsZ() -> Double {
        return readValue("tfs_z", legacyKeys: ["tfsZ"], defaultValue: defaultTfsZ)
    }

    func updateTfsZ(_ value: Double) {
        updateValue("tfs_z", value: value)
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
        return readValue("repetition_penalty", legacyKeys: ["penalty"], defaultValue: defaultPenalty)
    }

    func updatePenalty(_ value: Double) {
        updateValue("repetition_penalty", value: value)
    }

    // MARK: - N-gram

    func readNGram() -> Int {
        return readValue("n_gram", legacyKeys: ["nGram"], defaultValue: defaultNGram)
    }

    func updateNGram(_ value: Int) {
        updateValue("n_gram", value: value)
    }

    // MARK: - N-gram Factor

    func readNGramFactor() -> Double {
        return readValue("ngram_factor", legacyKeys: ["nGramFactor"], defaultValue: defaultNGramFactor)
    }

    func updateNGramFactor(_ value: Double) {
        updateValue("ngram_factor", value: value)
    }

    // MARK: - Read all config string

    func readConfigAsJSONString() -> String? {
        var mergedConfig: [String: Any] = [:]

        // Default config.json
        if let data = try? Data(contentsOf: configFileURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            mergedConfig = json
        }

        // Custom overrides
        if FileManager.default.fileExists(atPath: customConfigURL.path),
           let customData = try? Data(contentsOf: customConfigURL),
           var customJson = try? JSONSerialization.jsonObject(with: customData) as? [String: Any] {
            // Promote settings written by older app versions to the canonical
            // engine keys. Otherwise canonical keys in config.json win and the
            // gear UI appears to save values that the runtime never observes.
            let legacyKeyMap = [
                "topK": "top_k",
                "topP": "top_p",
                "minP": "min_p",
                "tfsZ": "tfs_z",
                "penalty": "repetition_penalty",
                "nGram": "n_gram",
                "nGramFactor": "ngram_factor",
            ]
            for (legacyKey, canonicalKey) in legacyKeyMap
                where customJson[canonicalKey] == nil && customJson[legacyKey] != nil {
                customJson[canonicalKey] = customJson[legacyKey]
            }
            mergedConfig.merge(customJson) { _, new in new }
        }

        guard !mergedConfig.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: mergedConfig, options: .prettyPrinted),
              let jsonString = String(data: data, encoding: .utf8)
        else {
            return nil
        }

        print("Config JSON String: \(jsonString)") // debug
        return jsonString
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
    case greedy
    case temperature
    case topK
    case topP
    case minP
    case tfs
    case typical
    case penalty
    case mixed

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
    case greedy
    case temperature

    var displayName: String {
        switch self {
        case .greedy: return "Greedy"
        case .temperature: return "Temperature"
        }
    }
}
