import Foundation

enum ContextBudget {
    static let defaultLimit = 4096
    static let maximumLimit = 65_536
    static let choices = [1024, 2048, 4096, 8192, 12_288, 16_384, 24_576, 32_768, 49_152, 65_536]
    static func clamp(_ value: Int, modelMaximum: Int?) -> Int {
        max(512, min(value, min(maximumLimit, modelMaximum ?? maximumLimit)))
    }
    static func promptBudget(limit: Int, maxOutput: Int) -> Int {
        limit - min(maxOutput, max(64, limit / 4))
    }
    struct MemoryProfile {
        let kvWidth: Int // Sum of KV heads * head dimension across full-attention layers.
        let kvLayers: Int
        let attentionHeads: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let linearWidth: Int
        let hasVision: Bool

        func kvBytesPerToken(usage: ContextUsage) -> Double {
            Double(kvWidth * (usage.kvKeyBytes + usage.kvValueBytes) + usage.kvScaleBytes * kvLayers)
        }

        func workspaceBytes(tokens: Int, usage: ContextUsage) -> Double {
            // Embeddings are FP32 on the host. Device buffers cover residuals,
            // projections, the gated MLP and (where present) linear attention.
            let activations = hiddenSize * 8 + intermediateSize * 3 + max(hiddenSize, linearWidth) * 4
            let analytic = Double(hiddenSize * 4 + activations * usage.activationBytes) * 1.25
            let perToken = max(analytic, usage.measuredWorkspaceBytesPerToken)
            let attention = Double(2 * attentionHeads * usage.activationBytes) * Double(tokens) * Double(tokens)
            // The bundled Metal backend splits Q rows to bound its two scratch
            // buffers to 128 MiB. Other backends use the unbounded estimate.
            let scratch = usage.metalBackend ? min(attention, 128 * 1024 * 1024) : attention
            return perToken * Double(tokens) + scratch
        }
    }

    struct Recommendation {
        let tokens: Int
        let kvBytesPerToken: Double
        let reserveBytes: Double
    }

    static func reserveBytes(profile: MemoryProfile, usage: ContextUsage) -> Double {
        let processLimit = Double(usage.availableMemoryBytes ?? 0) + Double(usage.footprintBytes ?? 0)
        return max(512 * 1024 * 1024, processLimit * 0.2) + (profile.hasVision ? 128 * 1024 * 1024 : 0)
    }

    /// os_proc_available_memory is the app's remaining dirty-memory allowance,
    /// already excluding loaded weights and allocations, rather than device RAM.
    static func recommendation(profile: MemoryProfile?, usage: ContextUsage, modelMaximum: Int?) -> Recommendation? {
        guard let profile, let available = usage.availableMemoryBytes, usage.footprintBytes != nil else { return nil }
        let kv = profile.kvBytesPerToken(usage: usage)
        let reserve = reserveBytes(profile: profile, usage: usage)
        let maximum = clamp(maximumLimit, modelMaximum: modelMaximum)
        let candidates = Array(Set([512, maximum] + choices)).filter { $0 <= maximum }.sorted()
        guard let tokens = candidates.last(where: {
            additionalBytes(tokens: $0, profile: profile, usage: usage) + reserve <= Double(available)
        }) else { return nil }
        return Recommendation(tokens: tokens, kvBytesPerToken: kv, reserveBytes: reserve)
    }

    static func additionalBytes(tokens: Int, prefillTokens: Int? = nil, profile: MemoryProfile, usage: ContextUsage) -> Double {
        let kv = profile.kvBytesPerToken(usage: usage)
        let cachedKV = kv * Double(usage.cachedTokens)
        // Reuse existing prefill allocations without crediting model weights.
        // Completed inferences refine the model baseline and workspace estimate.
        let reusableWorkspace = min(profile.workspaceBytes(tokens: usage.workspaceTokens, usage: usage),
            max(0, Double(usage.footprintBytes ?? 0) - Double(usage.modelBaselineBytes ?? usage.footprintBytes ?? 0) - cachedKV))
        let newKV = kv * Double(((tokens + 63) / 64) * 64)
        return max(0, newKV - cachedKV) * 1.15 + max(0,
            profile.workspaceBytes(tokens: prefillTokens ?? tokens, usage: usage) - reusableWorkspace)
    }

    static func memoryProfile(path: String) -> MemoryProfile? {
        var fields: [String: Any] = [:]
        for file in ["llm_config.json", "config.json", "configuration.json"] {
            guard let data = FileManager.default.contents(atPath: path + "/" + file),
                  let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            fields.merge(root["text_config"] as? [String: Any] ?? root) { _, new in new }
        }
        guard let hidden = fields["hidden_size"] as? Int, hidden > 0 else { return nil }
        let visual = fields["is_visual"] as? Bool ?? (fields["vision_config"] != nil)
        if fields["model_type"] as? String == "qwen3_5", hidden == 1024,
           fields["num_hidden_layers"] == nil {
            // This market package omits its attention dimensions. The original
            // 0.8B config has six full-attention and eighteen linear layers.
            // https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
            return MemoryProfile(kvWidth: 6 * 2 * 256, kvLayers: 6, attentionHeads: 8,
                hiddenSize: hidden, intermediateSize: 3584, linearWidth: 2048, hasVision: visual)
        }
        let shape = fields["key_value_shape"] as? [Int]
        guard let layers = fields["num_hidden_layers"] as? Int ?? fields["layer_nums"] as? Int,
              let headDim = fields["head_dim"] as? Int ?? shape?.last,
              let heads = fields["num_attention_heads"] as? Int ?? (headDim > 0 ? hidden / headDim : nil),
              let kvHeads = fields["num_key_value_heads"] as? Int ?? (shape?.count == 5 ? shape?[3] : nil),
              layers > 0, headDim > 0, heads > 0, kvHeads > 0 else { return nil }
        let layerTypes = fields["layer_types"] as? [String]
        let interval = fields["full_attention_interval"] as? Int ?? 1
        let fullLayers = layerTypes?.filter { !$0.contains("linear") }.count ?? layers / max(1, interval)
        let linearWidth = (fields["linear_num_key_heads"] as? Int ?? 0) * (fields["linear_key_head_dim"] as? Int ?? 0)
        return MemoryProfile(kvWidth: fullLayers * kvHeads * headDim, kvLayers: fullLayers,
            attentionHeads: heads, hiddenSize: hidden,
            intermediateSize: fields["intermediate_size"] as? Int ?? hidden * 4,
            linearWidth: linearWidth, hasVision: visual)
    }
    struct ModelLimit {
        let tokens: Int
        let fromOriginalModel: Bool
    }
    static func modelMaximum(path: String) -> Int? { modelLimit(path: path)?.tokens }
    static func modelLimit(path: String) -> ModelLimit? {
        for file in ["configuration.json", "config.json", "llm_config.json"] {
            guard let data = FileManager.default.contents(atPath: path + "/" + file),
                  let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            let text = root["text_config"] as? [String: Any] ?? root
            for key in ["max_position_embeddings", "context_length", "max_seq_len", "max_sequence_length"] {
                if let value = text[key] as? Int, value >= 512 {
                    return ModelLimit(tokens: value, fromOriginalModel: false)
                }
            }
        }
        // The published MNN package omits this field. This is the original
        // Qwen configuration's nominal value, not a verified mobile limit.
        // https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
        if URL(fileURLWithPath: path).lastPathComponent == "Qwen3.5-0.8B-MNN" {
            return ModelLimit(tokens: 262_144, fromOriginalModel: true)
        }
        return nil
    }
}

struct ContextUsage {
    var inputTokens = 0
    var generatedTokens = 0
    var budgetExhausted = false
    var availableMemoryBytes: UInt64?
    var footprintBytes: UInt64?
    var modelBaselineBytes: UInt64?
    var cachedTokens = 0
    var workspaceTokens = 0
    var kvKeyBytes = 2
    var kvValueBytes = 2
    var kvScaleBytes = 0
    var activationBytes = 2
    var metalBackend = true
    var measuredWorkspaceBytesPerToken = 0.0
    var totalTokens: Int { inputTokens + generatedTokens }
    func fraction(limit: Int) -> Double { min(1, max(0, Double(totalTokens) / Double(max(1, limit)))) }
}

struct ContextSummarySnapshot: Codable, Equatable {
    let modelId: String
    var summary: String
    var coveredTurns: Int
    // Version 2 counts unanswered requests too. Older derived summaries use
    // incompatible turn offsets; rebuild them from the preserved raw history.
    var formatVersion = 2
    var prompts: [[String: String]] {
        summary.isEmpty ? [] : [["system": "Earlier conversation summary (may omit details):\n" + summary]]
    }
    static func load(historyId: String, modelId: String) -> Self? {
        guard let data = try? Data(contentsOf: url(historyId)),
              let value = try? JSONDecoder().decode(Self.self, from: data),
              value.modelId == modelId, value.coveredTurns >= 0, value.formatVersion == 2 else { return nil }
        return value
    }
    func save(historyId: String) {
        let destination = Self.url(historyId)
        do {
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(self).write(to: destination, options: .atomic)
        } catch { /* Raw chat remains available if saving a derived summary fails. */ }
    }
    private static func url(_ historyId: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ChatHistories").appendingPathComponent(historyId)
            .appendingPathComponent("context-summary.json")
    }
}
