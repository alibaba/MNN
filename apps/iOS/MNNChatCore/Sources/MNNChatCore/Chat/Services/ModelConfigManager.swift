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

    private let defaultBackendType = "metal"
    private let defaultPrecision = "low"
    private let defaultThreadNum = 4
    private let defaultMaxNewTokens = 128
    private let maximumMaxNewTokens = 4096
    private let defaultTfsZ: Double = 1.0
    private let defaultTypical: Double = 1.0
    /// Engine default; `repetition_penalty` only takes effect above 1.0 (sampler.cpp).
    private let defaultPenalty: Double = 1.0
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

    /// Set the iOS default before a completed download is made available to chat.
    /// Keep explicit user overrides in custom_config.json separate from this default.
    func applyDownloadedModelDefaults() throws {
        try prepareTokenizerForInference()
        let data = try Data(contentsOf: configFileURL)
        guard var config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        config["backend_type"] = defaultBackendType
        let updated = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try updated.write(to: configFileURL, options: .atomic)
    }

    /// Migrate the old MiniCPM package default as well as its tokenizer before engine creation.
    /// Explicit choices in custom_config.json still take precedence.
    func prepareModelForInference() throws {
        try prepareTokenizerForInference()
        try prepareLegacyChatTemplate()
        guard URL(fileURLWithPath: modelPath).lastPathComponent.contains("MiniCPM4-0.5B"),
              FileManager.default.fileExists(atPath: configFileURL.path) else { return }
        let data = try Data(contentsOf: configFileURL)
        guard var config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        guard config["backend_type"] as? String == "cpu" else { return }
        config["backend_type"] = defaultBackendType
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: configFileURL, options: .atomic)
    }

    /// Current runtimes use Jinja; retain the old ChatML prefix and add per-message roles.
    private func prepareLegacyChatTemplate() throws {
        let file = URL(fileURLWithPath: modelPath).appendingPathComponent("llm_config.json")
        guard FileManager.default.fileExists(atPath: file.path),
              var config = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
        else { return }
        let userTemplate = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
        guard config["jinja"] == nil, let prompt = config["prompt_template"] as? String,
              prompt.hasSuffix(userTemplate) else { return }
        config["jinja"] = [
            "chat_template": "{% for message in messages %}" +
                "{% if loop.first and message['role'] != 'system' %}{{ legacy_prefix }}{% endif %}" +
                "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}" +
                "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
            "eos": "<|im_end|>",
            "context": ["legacy_prefix": String(prompt.dropLast(userTemplate.count))],
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            .write(to: file, options: .atomic)
    }

    /// Older MiniCPM4 packages lost the SentencePiece whitespace marker and normalizer flags.
    /// Repair the downloaded text vocabulary once; newer exports already include these flags.
    func prepareTokenizerForInference() throws {
        guard URL(fileURLWithPath: modelPath).lastPathComponent.contains("MiniCPM4-0.5B") else { return }
        let file = URL(fileURLWithPath: modelPath).appendingPathComponent("tokenizer.txt")
        guard FileManager.default.fileExists(atPath: file.path) else { return }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        let header = String(data: try handle.read(upToCount: 256) ?? Data(), encoding: .utf8) ?? ""
        let headerLines = header.components(separatedBy: "\n")
        guard headerLines.count > 3, headerLines[0].trimmingCharacters(in: .whitespaces) == "430 0",
              headerLines[3].split(separator: " ").count == 1 else { return }
        var lines = try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n")
        guard lines.count > 4, let vocabularySize = Int(lines[3]), lines.count >= vocabularySize + 4 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        // The original MiniCPM vocabulary retains one literal-space token. Its other
        // space-containing pieces use ▁, and its normalizer has add_dummy_prefix=false.
        for index in 4..<(vocabularySize + 4) {
            let fields = lines[index].split(separator: " ", maxSplits: 2)
            guard fields.count == 3, let data = Data(base64Encoded: String(fields[0])),
                  let piece = String(data: data, encoding: .utf8), piece.contains(" ") else { continue }
            if piece == " ", Float(fields[1]) == 0 { continue }
            let restored = piece.replacingOccurrences(of: " ", with: "▁")
            lines[index] = Data(restored.utf8).base64EncodedString() + " " + String(fields[1]) + " " + String(fields[2])
        }
        lines[3] += " 0 1"
        try Data(lines.joined(separator: "\n").utf8).write(to: file, options: .atomic)
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

    // MARK: - Thinking Mode

    /// Per-model thinking-mode override. `nil` means the user has not chosen
    /// and the model default applies (jinja.context.enable_thinking when
    /// present, otherwise on).
    func readEnableThinkingOverride() -> Bool? {
        guard FileManager.default.fileExists(atPath: customConfigURL.path),
              let data = try? Data(contentsOf: customConfigURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json["enable_thinking"] as? Bool
    }

    func updateEnableThinking(_ value: Bool) {
        updateValue("enable_thinking", value: value)
    }

    // App preference, separate from the engine's KV-cache configuration.
    func readConversationContextEnabled() -> Bool {
        readValue("conversation_context_enabled", defaultValue: false)
    }

    func updateConversationContextEnabled(_ value: Bool) {
        updateValue("conversation_context_enabled", value: value)
    }

    func readContextTokenLimit() -> Int {
        ContextBudget.clamp(readValue("context_token_limit", defaultValue: ContextBudget.defaultLimit),
                            modelMaximum: ContextBudget.modelMaximum(path: modelPath))
    }

    func updateContextTokenLimit(_ value: Int) { updateValue("context_token_limit", value: value) }
    func readAutoCompressContext() -> Bool { readValue("auto_compress_context", defaultValue: true) }
    func updateAutoCompressContext(_ value: Bool) { updateValue("auto_compress_context", value: value) }


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

        if mergedConfig["backend_type"] == nil {
            mergedConfig["backend_type"] = defaultBackendType
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

            // Route the persisted thinking-mode switch into the engine-visible
            // jinja context. The switch lives at the top level of
            // custom_config.json, but the engine only honors
            // jinja.context.enable_thinking — without this rewrite, every full
            // config push (e.g. settings sheet onDisappear) would silently
            // restore the model's own default and the toggle would look broken.
            if let thinkingOverride = customJson["enable_thinking"] as? Bool {
                var jinja = mergedConfig["jinja"] as? [String: Any] ?? [:]
                var context = jinja["context"] as? [String: Any] ?? [:]
                context["enable_thinking"] = thinkingOverride
                jinja["context"] = context
                mergedConfig["jinja"] = jinja
            }
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
        // Matches the engine default (llmconfig.hpp) when the model config omits it.
        let typeString = readValue("sampler_type", defaultValue: "mixed")
        return SamplerType(rawValue: typeString) ?? .mixed
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
    /// `repetition_penalty` is multiplicative and only takes effect above 1.0;
    /// MNN's docs recommend 1.05–1.5.
    static let penaltyRange: ClosedRange<Double> = 1.0 ... 1.5

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
