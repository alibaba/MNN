//
//  LLMChatViewModel.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/9/29.
//

import Foundation
import AVFoundation
import CMNNInference
import Combine
import SwiftUI
import UIKit

import ExyteChat

final class LLMChatViewModel: ObservableObject, StreamingMessageProvider {
    var llm: LLMInferenceEngineWrapper?
    private var diffusion: DiffusionSession?
    private var sanaDiffusion: SanaDiffusionSession?
    let llmState = LLMState()
    private var audioPlaybackManager: AudioPlaybackManager?
    private let presetASRService = PresetZipformerASRService()
    private var presetASRGeneration: UInt64 = 0
    private var isRecognizingPreset = false

    @Published var messages: [Message] = [] {
        didSet { historyRevision &+= 1 }
    }
    @Published private(set) var isStopping = false
    @Published private(set) var historySaveFailed = false
    private var historyRevision: UInt64 = 0
    private var savedHistoryRevision: UInt64 = 0
    private var pendingSaveRevision: UInt64?
    private var historySaveTimer: AnyCancellable?
    var isStarted = false
    private var modelLoadGeneration: UInt64 = 0
    private var requestThinkingOverride: Bool?
    // The ASR preset displays recognition metrics alongside its transcript.
    // Retrying it must use the original transcript rather than those metrics.
    private var lastRecognizedPrompt: (userId: String, transcript: String)?

    /// The streaming answer, rate-limited for rendering.
    ///
    /// The markdown renderer re-parses the entire answer on every change (its
    /// cache is keyed by the text itself, so a growing answer never hits it), so
    /// driving it per token costs one full parse per token — O(n²) across a
    /// response. That is why long answers visibly slow down as they grow, and
    /// why it gets worse once formulas start rendering: each re-parse also
    /// re-lays-out every formula produced so far. Decoupling redraw rate from
    /// decode rate keeps the cost linear; `latest: true` guarantees the final
    /// text still arrives.
    @Published var streamingAnswerText: String = ""

    @Published var isModelLoaded = false
    @Published var isProcessing: Bool = false
    @Published var loadFailed: Bool = false
    @Published var currentStreamingMessageId: String? = nil
    @Published var streamingStates: [String: StreamingMessageStateManager] = [:]
    private var pendingPerformanceData: [String: String] = [:]

    @Published var useMmap: Bool = false
    @Published var useMultimodalPromptAPI: Bool = true

    @Published private(set) var isConversationContextEnabled: Bool = false
    @Published private(set) var contextLimitTokens: Int = ContextBudget.defaultLimit
    @Published private(set) var autoCompressContext = true
    @Published private(set) var contextUsage = ContextUsage()
    @Published private(set) var isCompactingContext = false
    @Published var contextNotice: String?
    let modelContextLimit: ContextBudget.ModelLimit?
    let modelMemoryProfile: ContextBudget.MemoryProfile?
    var modelMaximumContextTokens: Int? { modelContextLimit?.tokens }
    private var contextSummary: ContextSummarySnapshot?
    var contextGeneration: UInt64 = 0
    var activeRequestGeneration: UInt64?
    private var contextUsageTimer: AnyCancellable?
    // A summary and a rebuilt prompt set are internal preparation work. Keep
    // the previous user request's usage until the next user stream is ready.
    var readsNativeContextUsage = true
    private var memorySampleID = 0
    private var memorySamples: [(input: Int, cache: Int, peak: Double)] = []
    private var measuredModelBaselineBytes: UInt64?
    private var measuredWorkspaceBytesPerToken = 0.0

    var contextRecommendation: ContextBudget.Recommendation? {
        ContextBudget.recommendation(profile: modelMemoryProfile, usage: contextUsage, modelMaximum: modelMaximumContextTokens)
    }
    var recommendedContextTokens: Int {
        contextRecommendation?.tokens ?? ContextBudget.clamp(ContextBudget.defaultLimit, modelMaximum: modelMaximumContextTokens)
    }
    var contextLimitChoices: [Int] {
        Array(Set([512, contextLimitTokens, recommendedContextTokens] + ContextBudget.choices))
            .filter { $0 <= ContextBudget.clamp(ContextBudget.maximumLimit, modelMaximum: modelMaximumContextTokens) }.sorted()
    }

    // MARK: - Think Mode Properties

    @Published var isThinkingModeEnabled: Bool = true
    /// Agent mode for this conversation (tool use: web search, page reading,
    /// time). Chosen before the first message and stored per conversation.
    @Published private(set) var isAgentMode = false
    @Published var supportsThinkingMode: Bool = false

    // MARK: - Sana Diffusion Default Prompt

    /// Default prompt for Sana Diffusion Ghibli style transfer
    static let sanaDiffusionDefaultPrompt = "Convert to a Ghibli-style illustration: soft contrast, warm tones, slight linework, keep the scene consistent."

    /// Default input text for the chat input field (used for Sana Diffusion default prompt)
    @Published var defaultInputText: String = ""

    var chatInputUnavilable: Bool {
        if isModelLoaded == false || isProcessing == true {
            return true
        }
        return false
    }

    var chatStatus: String {
        if isModelLoaded {
            if isProcessing {
                "Processing..."
            } else {
                "Ready"
            }
        } else {
            loadFailed ? "Load failed" : "Model Loading..."
        }
    }

    var chatCover: URL? {
        interactor.otherSenders.count == 1 ? interactor.otherSenders.first?.avatar : nil
    }

    let interactor: LLMChatInteractor
    private let showsModelLoadingMessages: Bool
    private var subscriptions = Set<AnyCancellable>()

    var modelInfo: ModelInfo
    var history: ChatHistory?
    private(set) var historyId: String
    var currentHistoryIdentifier: String { historyId }

    func savedConversation() -> ChatHistory? {
        saveHistoryImmediately()
        guard !historySaveFailed, messages.contains(where: { $0.user.isCurrentUser }) else { return nil }
        return ChatHistoryManager.shared.getAllHistory().first { $0.id == historyId }
    }

    let modelConfigManager: ModelConfigManager
    let supportsImageInput: Bool

    var isDiffusionModel: Bool {
        return modelInfo.name.lowercased().contains("stable-diffusion")
    }

    var isSanaDiffusionModel: Bool {
        return ModelUtils.isSanaDiffusionModel(modelInfo.name)
    }

    var isAnyDiffusionModel: Bool {
        return isDiffusionModel || isSanaDiffusionModel
    }

    init(modelInfo: ModelInfo, history: ChatHistory? = nil, showsModelLoadingMessages: Bool? = nil) {
        self.modelInfo = modelInfo
        supportsImageInput = ModelUtils.supportsImageInput(modelInfo)
        self.showsModelLoadingMessages = showsModelLoadingMessages ?? MNNChatConfiguration.current.showsModelLoadingBubble
        self.history = history
        historyId = history?.id ?? UUID().uuidString
        if let id = history?.id { isAgentMode = AgentStore.shared?.isAgentSession(id) ?? false }
        modelContextLimit = ContextBudget.modelLimit(path: modelInfo.localPath)
        modelMemoryProfile = ContextBudget.memoryProfile(path: modelInfo.localPath)
        contextSummary = ContextSummarySnapshot.load(historyId: historyId, modelId: modelInfo.id)
        let messages = self.history?.messages
        interactor = LLMChatInteractor(modelInfo: modelInfo, historyMessages: messages)

        modelConfigManager = ModelConfigManager(modelPath: modelInfo.localPath)

        useMmap = modelConfigManager.readUseMmap()
        useMultimodalPromptAPI = modelConfigManager.readUseMultimodalPromptAPI()
        isConversationContextEnabled = modelConfigManager.readConversationContextEnabled()
        contextLimitTokens = modelConfigManager.readContextTokenLimit()
        autoCompressContext = modelConfigManager.readAutoCompressContext()

        // Check if model supports thinking mode: either the classic tag gate,
        // or the model's config enables thinking (Qwen3.5-style jinja
        // templates that pre-open the think block).
        let thinkingEnabledByModelConfig: Bool = {
            guard let data = FileManager.default.contents(atPath: modelInfo.localPath + "/config.json"),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let jinja = json["jinja"] as? [String: Any],
                  let context = jinja["context"] as? [String: Any] else {
                return false
            }
            return context["enable_thinking"] as? Bool ?? false
        }()
        supportsThinkingMode = ModelUtils.isSupportThinkingSwitch(modelInfo.tags, modelName: modelInfo.name)
            || thinkingEnabledByModelConfig

        // A per-model settings override wins over the default-on behavior.
        if let override = modelConfigManager.readEnableThinkingOverride() {
            isThinkingModeEnabled = override
        }

        // Listen for streaming animation completion notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(onStreamingAnimationComplete(_:)),
            name: NSNotification.Name("StreamingAnimationCompleted"),
            object: nil
        )

        // Set up here rather than in onStart: onStart runs from onAppear and can
        // fire more than once, which would stack duplicate subscriptions.
        // Only the Pocket chat view consumes `streamingAnswerText`.
        // ~60fps. Low enough to keep the cost linear, high enough to still read
        // as a stream: at 20fps a fast decode lands ~30 tokens per update, which
        // reads as text jumping in chunks rather than flowing.
        if MNNChatConfiguration.current.streamsAnswerThrottled {
            $messages
                .map { $0.last { !$0.user.isCurrentUser }?.text ?? "" }
                .removeDuplicates()
                .throttle(for: .milliseconds(16), scheduler: DispatchQueue.main, latest: true)
                .sink { [weak self] text in
                    self?.streamingAnswerText = text
                }
                .store(in: &subscriptions)
        }
    }

    deinit {
        contextUsageTimer?.cancel()
        historySaveTimer?.cancel()
        presetASRGeneration &+= 1
        presetASRService.cancel()
        presetASRService.shutdown()
        // Cancel ongoing inference
        llm?.cancelInference()
        llm = nil
        isProcessing = false
        diffusion = nil
        
        // Stop audio playback
        audioPlaybackManager?.stop()
        audioPlaybackManager = nil

        sanaDiffusion = nil

        // Clean up streaming states
        clearAllStreamingStates()

        // Remove notification observers
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Think Mode Methods

    /// Toggle thinking mode on/off
    /// The mode can change only while the conversation has no messages.
    var canChangeAgentMode: Bool {
        !isProcessing && !isAnyDiffusionModel && !messages.contains(where: { $0.user.isCurrentUser })
    }

    func setAgentMode(_ enabled: Bool) {
        guard canChangeAgentMode else { return }
        isAgentMode = enabled
    }

    func toggleThinkingMode() {
        guard supportsThinkingMode, !isProcessing else { return }

        isThinkingModeEnabled.toggle()

        configureThinkingMode()

        // Persist per model so the choice survives app restarts and applies
        // at the next model load.
        modelConfigManager.updateEnableThinking(isThinkingModeEnabled)

        print("Think mode toggled to: \(isThinkingModeEnabled)")
    }

    func setupLLM(modelPath: String) {
        modelLoadGeneration &+= 1
        let generation = modelLoadGeneration
        isModelLoaded = false
        loadFailed = false
        if showsModelLoadingMessages {
            // Classic: loading progress is posted into the conversation as
            // system bubbles.
            Task { @MainActor in
                guard generation == self.modelLoadGeneration, !self.isModelLoaded, !self.loadFailed else { return }
                do {
                    try await self.send(draft: DraftMessage(
                        text: AppText.text("ModelLoadingText", comment: ""),
                        thinkText: "",
                        useMarkdown: false,
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ), userType: .system)
                } catch {
                    print("Error sending model loading status: \(error)")
                }
            }
        }

        if isSanaDiffusionModel {
            // Load Sana Diffusion model for style transfer
            sanaDiffusion = SanaDiffusionSession(modelPath: modelPath, completion: { [weak self] success in
                Task { @MainActor in
                    guard let self, generation == self.modelLoadGeneration else { return }
                    print("Sana Diffusion Model loaded: \(success)")
                    await self.sendModelLoadStatus(success: success)
                    guard generation == self.modelLoadGeneration else { return }
                    self.isModelLoaded = success
                    self.loadFailed = !success

                    // Set default prompt for Sana Diffusion
                    if success {
                        self.defaultInputText = LLMChatViewModel.sanaDiffusionDefaultPrompt
                    }
                }
            })
        } else if isDiffusionModel {
            diffusion = DiffusionSession(modelPath: modelPath, completion: { [weak self] success in
                Task { @MainActor in
                    guard let self, generation == self.modelLoadGeneration else { return }
                    print("Diffusion Model \(success)")
                    await self.sendModelLoadStatus(success: success)
                    guard generation == self.modelLoadGeneration else { return }
                    self.isModelLoaded = success
                    self.loadFailed = !success
                }
            })
        } else {
            do {
                try modelConfigManager.prepareModelForInference()
            } catch {
                loadFailed = true
                print("Error preparing model tokenizer: \(error)")
                return
            }
            llm = LLMInferenceEngineWrapper(modelPath: modelPath) { [weak self] success in
                Task { @MainActor in
                    guard let self, generation == self.modelLoadGeneration else { return }
                    if self.showsModelLoadingMessages {
                        await self.sendModelLoadStatus(success: success)
                    }
                    guard generation == self.modelLoadGeneration else { return }
                    self.isModelLoaded = success
                    self.loadFailed = !success

                    // Configure thinking mode after model is loaded
                    if success {
                        self.setModelConfig()
                        self.configureThinkingMode()
                        self.setupAudioOutput()
                        self.startContextUsageUpdates()
                    }
                }
            }
        }
    }

    /// Read-only access to the performance metadata attached to a message,
    /// used by the single-screen demo answer view.
    func pendingPerformanceDataReadable(for messageId: String) -> String? {
        interactor.performanceData(for: messageId)
    }

    /// Start an isolated conversation while keeping the previous one in history.
    func resetConversation() {
        guard !isProcessing else { return }

        saveHistoryImmediately()
        history = nil
        historyId = UUID().uuidString
        savedHistoryRevision = 0
        pendingSaveRevision = nil
        historySaveFailed = false
        lastRecognizedPrompt = nil
        contextSummary = nil
        contextNotice = nil
        readsNativeContextUsage = true
        contextUsage.inputTokens = 0
        contextUsage.generatedTokens = 0
        llm?.clearChatHistory()
        interactor.clearMessages()
        clearAllStreamingStates()
        messages = []
        // Cleared directly as well: the throttled pipeline would otherwise leave
        // the previous answer on screen for one interval.
        streamingAnswerText = ""
    }

    /// Shows a saved conversation with the model that is already loaded.
    /// Switching chats this way keeps the engine: no teardown, no reload.
    func loadConversation(_ target: ChatHistory) {
        guard !isProcessing else { return }
        saveHistoryImmediately()
        history = target
        historyId = target.id
        pendingSaveRevision = nil
        historySaveFailed = false
        lastRecognizedPrompt = nil
        contextSummary = ContextSummarySnapshot.load(historyId: target.id, modelId: modelInfo.id)
        contextNotice = nil
        readsNativeContextUsage = true
        contextUsage.inputTokens = 0
        contextUsage.generatedTokens = 0
        llm?.clearChatHistory()
        clearAllStreamingStates()
        streamingAnswerText = ""
        isAgentMode = AgentStore.shared?.isAgentSession(target.id) ?? false
        interactor.replaceMessages(with: target.messages)
        // The restored messages are what is stored; no checkpoint is needed.
        savedHistoryRevision = historyRevision
    }

    /// Configure thinking mode after model loading
    func configureThinkingMode() {
        guard let llm = llm, supportsThinkingMode else { return }

        let enabled = requestThinkingOverride ?? isThinkingModeEnabled
        llm.setThinkingModeEnabled(enabled)
        interactor.isThinkingModeEnabled = enabled

        print("Thinking mode configured: \(isThinkingModeEnabled)")
    }

    private func sendModelLoadStatus(success: Bool) async {
        let modelLoadSuccessText = AppText.text("ModelLoadingSuccessText", comment: "")
        let modelLoadFailText = AppText.text("ModelLoadingFailText", comment: "")
        let loadResult = success ? modelLoadSuccessText : modelLoadFailText

        do {
            try await send(draft: DraftMessage(
                text: loadResult,
                thinkText: "",
                useMarkdown: false,
                medias: [],
                recording: nil,
                replyMessage: nil,
                createdAt: Date()
            ), userType: .system)
        } catch {
            print("Error sending model load status: \(error)")
        }
    }

    func setConversationContextEnabled(_ enabled: Bool) {
        guard !isProcessing, enabled != isConversationContextEnabled else { return }
        isConversationContextEnabled = enabled
        modelConfigManager.updateConversationContextEnabled(enabled)
        llm?.setConversationContextEnabled(enabled)
        llm?.clearChatHistory()
    }

    func setContextTokenLimit(_ value: Int) {
        guard !isProcessing else { return }
        contextLimitTokens = ContextBudget.clamp(value, modelMaximum: modelMaximumContextTokens)
        modelConfigManager.updateContextTokenLimit(contextLimitTokens)
        llm?.setContextTokenLimit(contextLimitTokens)
    }

    func setAutoCompressContext(_ value: Bool) {
        guard !isProcessing else { return }
        autoCompressContext = value
        modelConfigManager.updateAutoCompressContext(value)
    }

    private func startContextUsageUpdates() {
        contextUsageTimer?.cancel()
        memorySampleID = 0
        memorySamples = []
        measuredModelBaselineBytes = nil
        measuredWorkspaceBytesPerToken = 0
        contextUsage.measuredWorkspaceBytesPerToken = 0
        contextUsage.modelBaselineBytes = nil
        llm?.setContextTokenLimit(contextLimitTokens)
        refreshContextUsage()
        contextUsageTimer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()
            .sink { [weak self] _ in
                guard let self, self.isProcessing else { return }
                self.refreshContextUsage()
            }
    }

    func refreshContextUsage() {
        guard let data = llm?.contextUsage() else { return }
        applyContextUsage(data)
    }

    func applyContextUsage(_ data: [String: NSNumber]) {
        var usage = contextUsage
        if readsNativeContextUsage && !isCompactingContext {
            usage.inputTokens = data["inputTokens"]?.intValue ?? 0
            usage.generatedTokens = data["generatedTokens"]?.intValue ?? 0
            usage.budgetExhausted = data["budgetExhausted"]?.boolValue ?? false
        }
        usage.availableMemoryBytes = data["availableMemoryBytes"]?.uint64Value
        usage.footprintBytes = data["footprintBytes"]?.uint64Value
        usage.cachedTokens = data["cachedTokens"]?.intValue ?? 0
        usage.workspaceTokens = data["workspaceTokens"]?.intValue ?? 0
        usage.kvKeyBytes = data["kvKeyBytes"]?.intValue ?? 2
        usage.kvValueBytes = data["kvValueBytes"]?.intValue ?? 2
        usage.kvScaleBytes = data["kvScaleBytes"]?.intValue ?? 0
        usage.activationBytes = data["activationBytes"]?.intValue ?? 2
        usage.metalBackend = data["metalBackend"]?.boolValue ?? true
        usage.measuredWorkspaceBytesPerToken = measuredWorkspaceBytesPerToken
        usage.modelBaselineBytes = measuredModelBaselineBytes ?? data["modelBaselineBytes"]?.uint64Value
        let sampleID = data["memorySampleID"]?.intValue ?? 0
        if sampleID != memorySampleID, let profile = modelMemoryProfile,
           let peak = data["memorySamplePeakBytes"]?.doubleValue,
           let input = data["memorySampleInputTokens"]?.intValue, input > 0 {
            memorySampleID = sampleID
            // A shorter request can retain buffers allocated by an earlier long
            // prefill. It must not be treated as a new model/weights baseline.
            guard memorySamples.isEmpty || input >= memorySamples.last!.input else { contextUsage = usage; return }
            let cache = data["memorySampleCacheTokens"]?.intValue ?? input
            memorySamples.append((input: input, cache: cache, peak: peak))
            memorySamples.sort { $0.input < $1.input }
            if memorySamples.count > 2 { memorySamples = [memorySamples.first!, memorySamples.last!] }
            let kv = profile.kvBytesPerToken(usage: usage)
            if let low = memorySamples.first, let high = memorySamples.last, high.input - low.input >= 512 {
                let growth = (high.peak - low.peak - kv * Double(high.cache - low.cache)) / Double(high.input - low.input)
                measuredWorkspaceBytesPerToken = max(measuredWorkspaceBytesPerToken, growth * 1.2)
                usage.measuredWorkspaceBytesPerToken = measuredWorkspaceBytesPerToken
            }
            let baseline = max(0, peak - kv * Double(cache) - profile.workspaceBytes(tokens: input, usage: usage))
            measuredModelBaselineBytes = max(measuredModelBaselineBytes ?? 0, UInt64(baseline))
            usage.modelBaselineBytes = measuredModelBaselineBytes
        }
        contextUsage = usage
    }

    private enum ContextPreparationError: Error, Equatable { case cancelled, unavailable, tooLong, summaryFailed, memory }

    @MainActor
    private func tokenCount(_ prompts: [[String: String]], engine: LLMInferenceEngineWrapper) async throws -> Int {
        let count = await withCheckedContinuation { continuation in
            engine.countTokens(for: prompts as [[AnyHashable: Any]]) { continuation.resume(returning: $0) }
        }
        guard count >= 0 else { throw ContextPreparationError.unavailable }
        return count
    }

    /// Snapshot at most once per checkpoint interval, never once per token.
    func saveHistoryCheckpoint() {
        guard messages.contains(where: { $0.user.isCurrentUser }),
              historyRevision > savedHistoryRevision, pendingSaveRevision == nil else { return }
        let revision = historyRevision
        let id = historyId
        pendingSaveRevision = revision
        ChatHistoryManager.shared.saveChatAsync(historyId: id, modelInfo: modelInfo, messages: messages) { [weak self] success in
            guard let self, id == self.historyId else { return }
            self.pendingSaveRevision = nil
            self.historySaveFailed = !success
            if success { self.savedHistoryRevision = max(self.savedHistoryRevision, revision) }
            // A reply may finish while its earlier checkpoint is still writing.
            if success && !self.isProcessing { self.saveHistoryCheckpoint() }
        }
    }

    /// Drain queued checkpoints before clearing temporary attachment files.
    func saveHistoryImmediately() {
        guard messages.contains(where: { $0.user.isCurrentUser }) else { return }
        contextSummary?.save(historyId: historyId)
        // Nothing changed since the last successful save: skip the synchronous
        // database write (this runs on the main thread when switching chats).
        if historyRevision == savedHistoryRevision, pendingSaveRevision == nil, !historySaveFailed { return }
        let success = ChatHistoryManager.shared.saveChat(historyId: historyId, modelInfo: modelInfo, messages: messages)
        historySaveFailed = !success
        if success { savedHistoryRevision = historyRevision }
    }

    func handleScenePhase(_ phase: ScenePhase) {
        if phase == .background { cancelGeneration() }
        if phase != .active { saveHistoryImmediately() }
    }

    func finishRequest() {
        activeRequestGeneration = nil
        isProcessing = false
        isStopping = false
        isCompactingContext = false
        if requestThinkingOverride != nil {
            requestThinkingOverride = nil
            configureThinkingMode()
        }
        saveHistoryCheckpoint()
    }

    @MainActor
    private func prepareConversationContext(content: String, generation: UInt64) async throws {
        guard let engine = llm else { throw ContextPreparationError.unavailable }
        refreshContextUsage()
        if let bytes = contextUsage.availableMemoryBytes, bytes < 128 * 1024 * 1024 {
            throw ContextPreparationError.memory
        }
        let promptBudget = ContextBudget.promptBudget(limit: contextLimitTokens, maxOutput: modelConfigManager.readMaxNewTokens())
        // A single oversized request cannot be made smaller by summarizing
        // history. Reject it before spending time on derived memory.
        guard try await tokenCount([["user": content]], engine: engine) <= promptBudget else {
            throw ContextPreparationError.tooLong
        }
        guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
        let turns = isConversationContextEnabled ? ConversationContextBuilder.turns(from: messages) : []
        if let snapshot = contextSummary, snapshot.coveredTurns > turns.count, isConversationContextEnabled {
            contextSummary = nil
        }
        var remaining = Array(turns.dropFirst(isConversationContextEnabled ? contextSummary?.coveredTurns ?? 0 : 0))
        var memory = isConversationContextEnabled ? contextSummary?.prompts ?? [] : []
        func request(_ recent: [ConversationContextBuilder.Turn], _ summary: [[String: String]]) -> [[String: String]] {
            summary + recent.flatMap(\.prompts) + [["user": content]]
        }
        var count = try await tokenCount(request(remaining, memory), engine: engine)
        if autoCompressContext, isConversationContextEnabled, count > promptBudget * 9 / 10, remaining.count > 2 {
            isCompactingContext = true
            do {
                try await compressContext(turns: turns, through: turns.count - 2, engine: engine, generation: generation)
                remaining = Array(turns.dropFirst(contextSummary?.coveredTurns ?? 0))
                memory = contextSummary?.prompts ?? []
                contextNotice = AppText.text("context.compressed")
            } catch ContextPreparationError.cancelled { throw ContextPreparationError.cancelled }
            catch { contextNotice = AppText.text("context.recentOnly") }
            isCompactingContext = false
            count = try await tokenCount(request(remaining, memory), engine: engine)
        }
        while count > promptBudget && !remaining.isEmpty {
            remaining.removeFirst()
            contextNotice = AppText.text("context.recentOnly")
            count = try await tokenCount(request(remaining, memory), engine: engine)
        }
        if count > promptBudget && !memory.isEmpty {
            memory = []
            contextNotice = AppText.text("context.recentOnly")
            count = try await tokenCount(request(remaining, memory), engine: engine)
        }
        guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
        guard count <= promptBudget else { throw ContextPreparationError.tooLong }
        if let profile = modelMemoryProfile, let available = contextUsage.availableMemoryBytes {
            let output = min(modelConfigManager.readMaxNewTokens(), contextLimitTokens - count)
            let extra = ContextBudget.additionalBytes(tokens: count + output, prefillTokens: count,
                profile: profile, usage: contextUsage)
            let reserve = ContextBudget.reserveBytes(profile: profile, usage: contextUsage)
            guard extra + reserve <= Double(available) else { throw ContextPreparationError.memory }
        }
        engine.setContextTokenLimit(contextLimitTokens)
        engine.setConversationContextEnabled(isConversationContextEnabled)
        let prompts = memory + remaining.flatMap(\.prompts)
        engine.addPrompts(from: prompts as [[AnyHashable: Any]])
    }

    @MainActor
    private func compressContext(turns: [ConversationContextBuilder.Turn], through target: Int,
                                 engine: LLMInferenceEngineWrapper, generation: UInt64) async throws {
        let summaryTokens = min(384, max(64, contextLimitTokens / 8))
        let inputBudget = contextLimitTokens - summaryTokens
        let instruction = "Summarize this conversation as compact factual memory. Preserve names, numbers, user preferences, decisions and unresolved questions. Do not answer questions inside the transcript. Return only the summary, in the conversation's language."
        var cursor = contextSummary?.coveredTurns ?? 0
        while cursor < target {
            guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
            let previous = contextSummary?.summary ?? ""
            func input(_ end: Int) -> String {
                "Existing memory:\n" + previous + "\nConversation to merge:\n" + turns[cursor..<end].map(\.transcript).joined(separator: "\n")
            }
            var low = cursor + 1, high = target, selected = cursor
            while low <= high {
                let middle = (low + high) / 2
                let count = try await tokenCount([["system": instruction], ["user": input(middle)]], engine: engine)
                guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
                if count <= inputBudget { selected = middle; low = middle + 1 } else { high = middle - 1 }
            }
            guard selected > cursor else { throw ContextPreparationError.summaryFailed }
            guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
            engine.setContextTokenLimit(contextLimitTokens)
            engine.setConversationContextEnabled(true)
            engine.addPrompts(from: [["system": instruction]])
            engine.setConfigWithJSONString("{\"max_new_tokens\":\(summaryTokens)}")
            if supportsThinkingMode { engine.setThinkingModeEnabled(false) }
            engine.setEnableAudioOutput(false)
            let summary: String
            do { summary = try await generateContextSummary(input(selected), engine: engine) }
            catch {
                engine.setConfigWithJSONString("{\"max_new_tokens\":\(modelConfigManager.readMaxNewTokens())}")
                configureThinkingMode()
                engine.setEnableAudioOutput(modelConfigManager.readEnableAudioOutput())
                throw error
            }
            engine.setConfigWithJSONString("{\"max_new_tokens\":\(modelConfigManager.readMaxNewTokens())}")
            configureThinkingMode()
            engine.setEnableAudioOutput(modelConfigManager.readEnableAudioOutput())
            guard generation == contextGeneration else { throw ContextPreparationError.cancelled }
            guard !summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ContextPreparationError.summaryFailed }
            contextSummary = ContextSummarySnapshot(modelId: modelInfo.id, summary: summary, coveredTurns: selected)
            contextSummary?.save(historyId: historyId)
            cursor = selected
        }
    }

    @MainActor
    private func generateContextSummary(_ input: String, engine: LLMInferenceEngineWrapper) async throws -> String {
        let result: String = try await withCheckedThrowingContinuation { continuation in
            var text = "", completed = false
            engine.processInput(input, withOutput: { output in
                guard !completed else { return }
                if output == "<eop>" || output == "<stoped>" {
                    completed = true
                    if output == "<stoped>" { continuation.resume(throwing: ContextPreparationError.cancelled) }
                    else if text.hasPrefix("Error:") { continuation.resume(throwing: ContextPreparationError.summaryFailed) }
                    else { continuation.resume(returning: text) }
                } else { text += output }
            })
        }
        while engine.isProcessing() { try await Task.sleep(for: .milliseconds(10)) }
        return result
    }

    /// Sends a draft message to the LLM for processing
    /// - Parameter draft: The draft message to send
    func sendToLLM(draft: DraftMessage) {
        guard !chatInputUnavilable else { return }
        isProcessing = !isAnyDiffusionModel
        isStopping = false
        contextGeneration &+= 1
        let generation = contextGeneration
        activeRequestGeneration = generation
        NotificationCenter.default.post(name: .dismissKeyboard, object: nil)

        Task { @MainActor in
            do {
                // Update Message UI and wait for completion
                guard activeRequestGeneration == generation, isStarted else { return }
                try await send(draft: draft, userType: .user)
                saveHistoryCheckpoint()
                guard generation == contextGeneration, isStarted else {
                    if activeRequestGeneration == generation { finishRequest() }
                    return
                }

                recordModelUsage()

                if isModelLoaded {
                    if isSanaDiffusionModel {
                        getSanaDiffusionResponse(draft: draft)
                    } else if isDiffusionModel {
                        getDiffusionResponse(draft: draft)
                    } else if isAgentMode, draft.medias.isEmpty, draft.recording == nil,
                              !draft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        AgentStore.shared?.setAgentSession(historyId, enabled: true)
                        await runAgentRequest(draft.text, generation: generation)
                    } else {
                        getLLMRespsonse(draft: draft, generation: generation)
                    }
                }
            } catch {
                finishRequest()
                print("Error sending message to LLM: \(error)")
                // Send error message to user
                Task {
                    do {
                        try await send(draft: DraftMessage(
                            text: "Error: Failed to send message. Please try again.",
                            thinkText: "",
                            useMarkdown: false,
                            medias: [],
                            recording: nil,
                            replyMessage: nil,
                            createdAt: Date()
                        ), userType: .system)
                    } catch {
                        print("Failed to send error message: \(error)")
                    }
                }
            }
        }
    }

    /// Sends a draft message to the chat interactor asynchronously
    /// - Parameters:
    ///   - draft: The draft message to send
    ///   - userType: The type of user sending the message
    /// - Throws: Any error that occurs during message sending
    func send(draft: DraftMessage, userType: UserType) async throws {
        try await interactor.send(draftMessage: draft, userType: userType)
    }

    func getDiffusionResponse(draft: DraftMessage) {
        Task {
            let tempImagePath = FileOperationManager.shared.generateTempImagePath().path

            var lastProcess: Int32 = 0

            try await self.send(draft: DraftMessage(text: "Start Generating Image...", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .assistant)

            // Get user-configured iteration count and seed value
            let userIterations = self.modelConfigManager.readIterations()
            let userSeed = self.modelConfigManager.readSeed()

            diffusion?.run(withPrompt: draft.text,
                           imagePath: tempImagePath,
                           iterations: Int32(userIterations),
                           seed: Int32(userSeed),
                           progressCallback: { [weak self] progress in
                               guard let self = self else { return }
                               if progress == 100 {
                                   Task {
                                       do {
                                           try await self.send(draft: DraftMessage(text: "Image generated successfully!", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                                       } catch {
                                           print("Error sending image generation success message: \(error)")
                                       }
                                   }
                                   self.interactor.sendImage(imageURL: URL(fileURLWithPath: tempImagePath))
                               } else if (progress - lastProcess) > 20 {
                                   lastProcess = progress
                                   Task {
                                       do {
                                           try await self.send(draft: DraftMessage(text: "Generating Image \(progress)%", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                                       } catch {
                                           print("Error sending image generation progress message: \(error)")
                                       }
                                   }
                               }
                           })
        }
    }

    // MARK: - Sana Diffusion Style Transfer

    func getSanaDiffusionResponse(draft: DraftMessage) {
        Task {
            // 1. Check if we have an input image
            guard !draft.medias.isEmpty else {
                try? await self.send(
                    draft: DraftMessage(
                        text: AppText.text("Please select an image for style transfer.", comment: ""),
                        thinkText: "",
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ),
                    userType: .system
                )
                return
            }

            // 2. Get the first image from medias
            var inputImagePath: String?
            for media in draft.medias {
                guard media.type == .image, let url = await media.getURL() else {
                    continue
                }

                let fileName = url.lastPathComponent
                if let processedUrl = FileOperationManager.shared.processImageFile(from: url, fileName: fileName, requiresJPEG: true) {
                    inputImagePath = processedUrl.path
                    break
                }
            }

            guard let inputPath = inputImagePath else {
                try? await self.send(
                    draft: DraftMessage(
                        text: AppText.text("Unsupported image format. Please use JPG/JPEG images for style transfer.", comment: ""),
                        thinkText: "",
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ),
                    userType: .system
                )
                return
            }

            // 3. Prepare output path
            let outputPath = FileOperationManager.shared.generateTempImagePath().path

            // 4. Get prompt (use default if empty)
            let prompt = draft.text.isEmpty ? LLMChatViewModel.sanaDiffusionDefaultPrompt : draft.text

            // 5. Get user-configured iteration count and seed value
            let userIterations = self.modelConfigManager.readIterations()
            let userSeed = self.modelConfigManager.readSeed()

            // 6. Show initial status (one system message; we will update it in place for progress)
            let initialStage = AppText.text("Starting style transfer...", comment: "")
            try? await self.send(
                draft: DraftMessage(
                    text: "\(initialStage) (0%)",
                    thinkText: "",
                    useMarkdown: false,
                    medias: [],
                    recording: nil,
                    replyMessage: nil,
                    createdAt: Date()
                ),
                userType: .system
            )

            await MainActor.run {
                self.isProcessing = true
            }

            var lastProgress: Int32 = 0
            // 7. Run style transfer
            sanaDiffusion?.runStyleTransfer(
                withInputImage: inputPath,
                prompt: prompt,
                outputPath: outputPath,
                iterations: Int32(userIterations),
                seed: Int32(userSeed),
                progressCallback: { [weak self] progress, _ in
                    guard let self = self else { return }
                    if progress >= 100 || progress - lastProgress >= 5 {
                        lastProgress = progress
                        let stageText: String
                        if progress <= 10 {
                            stageText = AppText.text("Processing prompt...", comment: "Sana diffusion progress stage")
                        } else if progress >= 95 {
                            stageText = AppText.text("Generating image...", comment: "Sana diffusion progress stage")
                        } else {
                            stageText = AppText.text("Running diffusion...", comment: "Sana diffusion progress stage")
                        }
                        self.interactor.updateLastMessage(text: "\(stageText) (\(progress)%)")
                    }
                },
                completion: { [weak self] success, error, totalTimeMs in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.isProcessing = false

                        if success {
                            let completionText = AppText.text("Style transfer completed!", comment: "")
                            self.interactor.updateLastMessage(text: completionText)
                            
                            // Send total time as a separate message after the image
                            let totalTimeSec = totalTimeMs / 1000.0
                            let timeText = String(format: "%.1f", totalTimeSec)
                            let timeMessage = AppText.text("Total time:", comment: "Sana diffusion total time label") + " \(timeText)s"
                            do {
                                try await self.send(draft: DraftMessage(text: timeMessage, thinkText: "", useMarkdown: false, medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                            } catch {
                                print("Error sending time message: \(error)")
                            }
                            
                            self.interactor.sendImage(imageURL: URL(fileURLWithPath: outputPath))
                            
                        } else {
                            let errorMessage = error ?? AppText.text("Style transfer failed.", comment: "")
                            self.interactor.updateLastMessage(text: errorMessage)
                        }
                    }
                }
            )
        }
    }

    func getLLMRespsonse(draft: DraftMessage, generation: UInt64? = nil) {
        Task { @MainActor in
            await llmState.setProcessing(true)
            var content = draft.text
            let medias = draft.medias
            var multimodalImagePlaceholders: [String] = []
            var legacyImagePlaceholders: [String] = []
            var videoPlaceholders: [String] = []
            var imageDictionary: [String: UIImage] = [:]
            var missingAttachments: [String] = []
            var hasVideoInput = false
            let shouldUseMultimodalAPI = self.useMultimodalPromptAPI

            for (index, media) in medias.enumerated() {
                switch media.type {
                case .image:
                    guard let url = await media.getURL() else { continue }
                    let fileName = media.id.uuidString + "." + url.pathExtension

                    guard let processedUrl = FileOperationManager.shared.processImageFile(from: url, fileName: fileName),
                          FileOperationManager.shared.fileExists(at: processedUrl) else {
                        missingAttachments.append("图片 \(fileName) 无法读取，已跳过。")
                        continue
                    }

                    if shouldUseMultimodalAPI {
                        let key = "img_\(index)"
                        guard let image = UIImage(contentsOfFile: processedUrl.path) else {
                            missingAttachments.append("图片 \(fileName) 转换失败，已跳过。")
                            continue
                        }
                        imageDictionary[key] = image
                        multimodalImagePlaceholders.append("<img>\(key)</img>")
                    } else {
                        legacyImagePlaceholders.append("<img>\(processedUrl.path)</img>")
                    }
                case .video:
                    guard let url = await media.getURL() else { continue }
                    let fileName = url.lastPathComponent
                    guard let preparedURL = FileOperationManager.shared.prepareVideoFileURL(from: url, fileName: fileName) else {
                        missingAttachments.append("视频 \(fileName) 复制失败，已跳过。")
                        continue
                    }
                    guard FileOperationManager.shared.fileExists(at: preparedURL) else {
                        missingAttachments.append("视频 \(fileName) 文件不存在或已被移除。")
                        continue
                    }
                    videoPlaceholders.append("<video>\(preparedURL.path)</video>")
                    hasVideoInput = true
                default:
                    continue
                }
            }

            let selectedImagePlaceholders = shouldUseMultimodalAPI ? multimodalImagePlaceholders : legacyImagePlaceholders
            if !selectedImagePlaceholders.isEmpty || !videoPlaceholders.isEmpty {
                let mediaPrefix = (selectedImagePlaceholders + videoPlaceholders).joined()
                content = mediaPrefix + content
            }

            if let audio = draft.recording, let path = audio.url {
                if FileOperationManager.shared.fileExists(at: path) {
                    content = "<audio>\(path.path)</audio>" + content
                } else {
                    missingAttachments.append("音频文件已丢失，未能发送。")
                }
            }

            if !missingAttachments.isEmpty {
                let warningDraft = DraftMessage(
                    text: missingAttachments.joined(separator: "\n"),
                    thinkText: "",
                    medias: [],
                    recording: nil,
                    replyMessage: nil,
                    createdAt: Date()
                )
                do {
                    try await self.send(draft: warningDraft, userType: .system)
                } catch {
                    print("Error sending missing attachment warning: \(error)")
                }
            }

            let hasImageInput = shouldUseMultimodalAPI ? !imageDictionary.isEmpty : !legacyImagePlaceholders.isEmpty
            let hasAudioInput = draft.recording != nil && FileOperationManager.shared.fileExists(at: draft.recording?.url)
            let hasVisualInput = hasImageInput || hasVideoInput
            let hasTextInput = !draft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

            if !hasTextInput && (hasVisualInput || hasAudioInput) {
                let defaultPrompt = modelConfigManager.readDefaultMultimodalPrompt()
                if !defaultPrompt.isEmpty {
                    content = defaultPrompt + "\n" + content
                }
            }

            if !hasVisualInput && !hasAudioInput && !hasTextInput {
                await llmState.setProcessing(false)
                await MainActor.run { self.finishRequest() }
                let warningText = AppText.text(
                    "video.frameExtractionFailed",
                    comment: "Warning shown when a pure video input cannot provide frames."
                )
                let warningDraft = DraftMessage(
                    text: warningText,
                    thinkText: "",
                    useMarkdown: false,
                    medias: [],
                    recording: nil,
                    replyMessage: nil,
                    createdAt: Date()
                )
                do {
                    try await self.send(draft: warningDraft, userType: .system)
                } catch {
                    print("Error sending warning message: \(error)")
                }
                return
            }

            await runLLMInference(content: content, images: imageDictionary, useMultimodalAPI: shouldUseMultimodalAPI,
                                  expectedGeneration: generation)
        }
    }

    /// Runs streaming inference for already-prepared content and appends the result to the chat.
    /// Shared by the normal send flow and preset prompts.
    func runLLMInference(content: String, images: [String: UIImage], useMultimodalAPI: Bool,
                         expectedGeneration: UInt64? = nil) async {
        if let expectedGeneration, expectedGeneration != activeRequestGeneration { return }
        await llmState.setProcessing(true)
        let generation = await MainActor.run { () -> UInt64 in
            self.isProcessing = true
            self.contextNotice = nil
            self.readsNativeContextUsage = false
            if expectedGeneration == nil { self.activeRequestGeneration = self.contextGeneration }
            return expectedGeneration ?? self.contextGeneration
        }
        let loadGeneration = modelLoadGeneration
        guard let engine = llm, generation == contextGeneration, isStarted else {
            await llmState.setProcessing(false)
            await MainActor.run {
                if self.activeRequestGeneration == generation { self.finishRequest() }
            }
            return
        }
        do { try await prepareConversationContext(content: content, generation: generation) }
        catch {
            await llmState.setProcessing(false)
            await MainActor.run {
                if self.activeRequestGeneration == generation { self.finishRequest() }
            }
            if generation == contextGeneration {
                let key: String
                switch error as? ContextPreparationError {
                case .memory: key = "context.error.memory"
                case .tooLong: key = "context.error.tooLong"
                default: key = "context.error.unavailable"
                }
                try? await send(draft: DraftMessage(text: AppText.text(key), thinkText: "", medias: [],
                                                   recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
            }
            return
        }
        guard activeRequestGeneration == generation, isStarted else { return }

        // First, send the empty message asynchronously
        let emptyMessage = DraftMessage(
            text: "",
            thinkText: "",
            useMarkdown: true,
            medias: [],
            recording: nil,
            replyMessage: nil,
            createdAt: Date()
        )

        do {
            try await self.send(draft: emptyMessage, userType: .assistant)
        } catch {
            print("Error sending empty message: \(error)")
            await llmState.setProcessing(false)
            await MainActor.run { self.finishRequest() }
            return
        }

        // Then update UI state on main actor
        await MainActor.run {
            self.isProcessing = true
            if let lastMessage = self.messages.last {
                self.currentStreamingMessageId = lastMessage.id

                // Create and start state manager
                let stateManager = StreamingMessageStateManager(messageId: lastMessage.id)
                self.streamingStates[lastMessage.id] = stateManager
                stateManager.startStreaming()
            }
        }

        let convertedContent = content
        let (outputStream, outputContinuation) = AsyncStream.makeStream(of: String.self)

        // Consume native callbacks through one task so response fragments,
        // performance statistics, and completion are applied in FIFO order.
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            var pendingPerformanceData: String?
            var pendingOutput = ""
            var lastPresentation = ProcessInfo.processInfo.systemUptime
            var usageIsReady = false
            // Group display updates so long thinking text is not laid out per token.
            // Native fragments remain ordered, and completion flushes the final group.
            let presentationInterval: TimeInterval = 0.08
#if DEBUG
            let logAutoPresetResponse = CommandLine.arguments.contains("--auto-preset")
            var autoPresetResponse = ""
#endif

            for await output in outputStream {
                guard loadGeneration == self.modelLoadGeneration, self.isStarted else { break }
                if !usageIsReady {
                    usageIsReady = true
                    await MainActor.run {
                        self.readsNativeContextUsage = true
                        self.refreshContextUsage()
                    }
                }
                let performancePrefix = "<performance>"
                if output.hasPrefix(performancePrefix) {
                    let performanceData = String(output.dropFirst(performancePrefix.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !performanceData.isEmpty {
                        // Keep metrics hidden until the model has emitted its
                        // completion marker, so they always follow the answer.
                        pendingPerformanceData = performanceData
                    }
                    continue
                }

                if output == "<eop>" || output == "<stoped>" {
                    if !pendingOutput.isEmpty {
                        do {
                            try await self.send(draft: DraftMessage(
                                text: pendingOutput,
                                thinkText: "",
                                performanceData: nil,
                                useMarkdown: true,
                                medias: [],
                                recording: nil,
                                replyMessage: nil,
                                createdAt: Date()
                            ), userType: .assistant)
                        } catch {
                            print("Error sending final streaming message: \(error)")
                        }
                        pendingOutput = ""
                    }
#if DEBUG
                    if logAutoPresetResponse {
                        #if DEBUG
                        print("[AutoPreset] response: \(autoPresetResponse)")
                        #endif
                    }
#endif
                    let completedPerformanceData = pendingPerformanceData
                    await MainActor.run {
                        if let messageId = self.currentStreamingMessageId {
                            if let completedPerformanceData {
                                self.pendingPerformanceData[messageId] = completedPerformanceData
                                // Attach perf data to the message right away: the single-screen
                                // demo UI reads it from the message model instead of waiting for
                                // the bubble-stream animation completion gate below.
                                self.interactor.updatePerformanceData(completedPerformanceData, for: messageId)
                            }
                            if let stateManager = self.streamingStates[messageId] {
                                stateManager.markOutputComplete()
                                self.streamingStates[messageId] = stateManager
                            }

                            // Deliver completion directly to the visible cell. The
                            // provider is stored as an environment reference, so a
                            // state-only change does not always invalidate that cell.
                            NotificationCenter.default.post(
                                name: NSNotification.Name("StreamingOutputCompleted"),
                                object: nil,
                                userInfo: ["messageId": messageId]
                            )

                            // A recycled/off-screen cell cannot report animation
                            // completion. Keep a generous presentation-only fallback;
                            // inference and input controls are already complete below.
                            let characterCount = self.messages.first(where: { $0.id == messageId })?.text.count ?? 0
                            let fallbackDelay = min(max(Double(characterCount) * 0.02 + 2.0, 5.0), 30.0)
                            DispatchQueue.main.asyncAfter(deadline: .now() + fallbackDelay) { [weak self] in
                                self?.finishStreamingPresentation(for: messageId)
                            }
                        }

                        // Input availability follows native inference, never a view
                        // animation callback. This guarantees the app cannot deadlock.
                        self.refreshContextUsage()
                        if output == "<stoped>" {
                            self.contextNotice = AppText.text("chat.generationStopped")
                        } else if self.contextUsage.budgetExhausted {
                            self.contextNotice = AppText.text("context.limitReached")
                        } else if let message = self.messages.first(where: { $0.id == self.currentStreamingMessageId }),
                                  message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                  !(message.thinkText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.contextNotice = AppText.text("context.noFinalResponse")
                        }
                        self.finishRequest()

                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            NotificationCenter.default.post(name: .dismissKeyboard, object: nil)
                        }
                    }
                    await self.llmState.setProcessing(false)
                    break
                }

#if DEBUG
                if logAutoPresetResponse {
                    autoPresetResponse += output
                }
#endif
                pendingOutput += output.hasPrefix("Error: Input exceeds the context budget.")
                    ? AppText.text("context.error.tooLong") : output
                guard ProcessInfo.processInfo.systemUptime - lastPresentation >= presentationInterval else { continue }
                let presentationOutput = pendingOutput
                pendingOutput = ""
                lastPresentation = ProcessInfo.processInfo.systemUptime
                do {
                    try await self.send(draft: DraftMessage(
                        text: presentationOutput,
                        thinkText: "",
                        performanceData: nil,
                        useMarkdown: true,
                        medias: [],
                        recording: nil,
                        replyMessage: nil,
                        createdAt: Date()
                    ), userType: .assistant)
                } catch {
                    print("Error sending streaming message: \(error)")
                }
            }
        }

        let outputHandler: (String) -> Void = { output in
            outputContinuation.yield(output)
            if output == "<eop>" || output == "<stoped>" {
                outputContinuation.finish()
            }
        }

        // Start on the main actor without a suspension between the cancellation
        // guard and native dispatch, so a stop during preparation cannot restart it.
        let launched = await MainActor.run { () -> Bool in
            guard generation == self.contextGeneration, loadGeneration == self.modelLoadGeneration, self.isStarted else {
                if self.activeRequestGeneration == generation {
                    self.clearAllStreamingStates()
                    self.finishRequest()
                }
                outputContinuation.finish()
                return false
            }
            if useMultimodalAPI {
                engine.processMultimodalInput(convertedContent, images: images, withOutput: outputHandler, showPerformance: true)
            } else {
                engine.processInput(convertedContent, withOutput: outputHandler, showPerformance: true)
            }
            return true
        }
        if !launched { await llmState.setProcessing(false) }
    }

    /// Sends a preset prompt (optionally with a bundled image) through the chat pipeline.
    func sendPreset(_ preset: PresetPrompt) {
        guard !chatInputUnavilable else { return }

        if let audioPath = preset.audioBundlePath,
           let expectedHash = preset.audioSHA256
        {
            presetASRGeneration &+= 1
            let generation = presetASRGeneration
            isProcessing = true
            isRecognizingPreset = true
            isStopping = false
            contextGeneration &+= 1
            let requestGeneration = contextGeneration
            activeRequestGeneration = requestGeneration
            let audioURL = URL(fileURLWithPath: audioPath)

            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    let recognition = try await self.presetASRService.recognize(
                        audioURL: audioURL,
                        expectedAudioSHA256: expectedHash
                    )
                    guard generation == self.presetASRGeneration else { return }
                    self.isRecognizingPreset = false
                    await self.interactor.sendPresetAudioMessage(
                        text: recognition.displayText,
                        recording: recognition.recording,
                        expectsResponse: true
                    )
                    if let user = self.messages.last(where: { $0.user.isCurrentUser }) {
                        self.lastRecognizedPrompt = (user.id, recognition.transcript)
                    }
                    self.saveHistoryCheckpoint()
                    guard generation == self.presetASRGeneration else { return }
                    self.isRecognizingPreset = false
                    await self.runLLMInference(
                        content: recognition.transcript,
                        images: [:],
                        useMultimodalAPI: false,
                        expectedGeneration: requestGeneration
                    )
                } catch {
                    guard generation == self.presetASRGeneration else { return }
                    self.isRecognizingPreset = false
                    let failure = AppText.text("ASR 识别失败：", comment: "Preset ASR failure")
                        + error.localizedDescription
                    let recording = Recording(duration: 0, waveformSamples: [], url: audioURL)
                    await self.interactor.sendPresetAudioMessage(
                        text: failure,
                        recording: recording,
                        expectsResponse: false
                    )
                    await MainActor.run {
                        self.finishRequest()
                    }
                }
            }
            return
        }

        if let imagePath = preset.imageBundlePath {
            isProcessing = true
            isStopping = false
            contextGeneration &+= 1
            let generation = contextGeneration
            activeRequestGeneration = generation
            let imageURL = URL(fileURLWithPath: imagePath)
            let content = "<img>\(imagePath)</img>" + preset.text
            Task { @MainActor in
                await interactor.sendPresetMessage(text: preset.text, imageURL: imageURL)
                await MainActor.run { self.saveHistoryCheckpoint() }
                await runLLMInference(content: content, images: [:], useMultimodalAPI: true, expectedGeneration: generation)
            }
        } else {
            sendToLLM(draft: DraftMessage(
                text: preset.text,
                thinkText: nil,
                medias: [],
                recording: nil,
                replyMessage: nil,
                createdAt: Date()
            ))
        }
    }

    /// Sends a prompt using the selected conversation-context mode.
    func sendUserPrompt(_ text: String) {
        guard !chatInputUnavilable else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        sendToLLM(draft: DraftMessage(
            text: trimmed,
            thinkText: nil,
            medias: [],
            recording: nil,
            replyMessage: nil,
            createdAt: Date()
        ))
    }

    var canRegenerate: Bool {
        isModelLoaded && !isAnyDiffusionModel && !isProcessing && messages.contains(where: { $0.user.isCurrentUser })
    }

    var canGenerateDirectAnswer: Bool {
        guard canRegenerate, supportsThinkingMode,
              let userIndex = messages.lastIndex(where: { $0.user.isCurrentUser }),
              let answer = messages.dropFirst(userIndex + 1).last(where: { $0.user.id == "2" }) else { return false }
        return answer.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !(answer.thinkText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The user message an answer replies to.
    func promptMessage(for answerId: String) -> Message? {
        guard let index = messages.firstIndex(where: { $0.id == answerId }) else { return nil }
        return messages[..<index].last(where: { $0.user.isCurrentUser })
    }

    /// Text-only questions can be edited and sent again.
    func canEditAndResend(_ message: Message) -> Bool {
        canRegenerate && message.user.isCurrentUser && message.attachments.isEmpty && message.recording == nil
    }

    /// Replaces a question and everything after it with the edited text.
    func editAndResend(_ messageId: String, text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let message = messages.first(where: { $0.id == messageId }),
              canEditAndResend(message) else { return }
        clearAllStreamingStates()
        guard interactor.truncate(from: messageId) else { return }
        sendUserPrompt(trimmed)
    }

    /// Rebuild the same question, excluding the answer being replaced. With
    /// `userMessageId`, later turns are dropped and that question is answered again.
    func regenerateLastResponse(directAnswer: Bool = false, userMessageId: String? = nil) {
        let target = userMessageId.flatMap { id in messages.first { $0.id == id && $0.user.isCurrentUser } }
            ?? messages.last(where: { $0.user.isCurrentUser })
        guard canRegenerate, !directAnswer || canGenerateDirectAnswer, let user = target else { return }
        var content = user.text
        var images: [String: UIImage] = [:]
        var prefix = ""
        for (index, attachment) in user.attachments.enumerated() {
            guard FileManager.default.fileExists(atPath: attachment.full.path) else {
                contextNotice = AppText.text("chat.retryMissingAttachment")
                return
            }
            if attachment.type == .image {
                guard let imageURL = FileOperationManager.shared.processImageFile(
                    from: attachment.full, fileName: attachment.id + "." + attachment.full.pathExtension) else {
                    contextNotice = AppText.text("chat.retryMissingAttachment")
                    return
                }
                if useMultimodalPromptAPI {
                    guard let image = UIImage(contentsOfFile: imageURL.path) else {
                        contextNotice = AppText.text("chat.retryMissingAttachment")
                        return
                    }
                    let key = "img_\(index)"
                    images[key] = image
                    prefix += "<img>\(key)</img>"
                } else { prefix += "<img>\(imageURL.path)</img>" }
            } else if attachment.type == .video {
                prefix += "<video>\(attachment.full.path)</video>"
            }
        }
        if let recording = user.recording {
            if let recognized = lastRecognizedPrompt, recognized.userId == user.id {
                content = recognized.transcript
            } else if ModelUtils.isAudioModel(modelInfo.name) || modelInfo.tags.contains(where: { $0.localizedCaseInsensitiveContains("audio") }),
                      let url = recording.url, FileManager.default.fileExists(atPath: url.path) {
                prefix += "<audio>\(url.path)</audio>"
            } else {
                contextNotice = AppText.text("chat.retryAudioPrompt")
                return
            }
        }
        if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !prefix.isEmpty {
            content = modelConfigManager.readDefaultMultimodalPrompt() + "\n"
        }
        guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !prefix.isEmpty else { return }
        contextGeneration &+= 1
        let generation = contextGeneration
        activeRequestGeneration = generation
        isProcessing = true
        isStopping = false
        if directAnswer { requestThinkingOverride = false; configureThinkingMode() }
        clearAllStreamingStates()
        Task { @MainActor in
            guard activeRequestGeneration == generation, isStarted else { return }
            guard interactor.beginRegeneration(after: user.id) else { finishRequest(); return }
            saveHistoryCheckpoint()
            if isAgentMode, user.attachments.isEmpty, user.recording == nil {
                await runAgentRequest(content, generation: generation)
                return
            }
            await runLLMInference(content: prefix + content, images: images, useMultimodalAPI: useMultimodalPromptAPI,
                                  expectedGeneration: generation)
        }
    }

    func cancelGeneration() {
        guard isProcessing, !isStopping, !isAnyDiffusionModel else { return }
        isStopping = true
        contextGeneration &+= 1
        presetASRGeneration &+= 1
        presetASRService.cancel()
        llm?.cancelInference()
        // Recognition has no native LLM terminal marker to finish the request.
        if isRecognizingPreset {
            isRecognizingPreset = false
            finishRequest()
        }
    }

    /// Retrieves batch LLM responses for the provided prompts.
    ///
    /// This method forwards the prompts to the LLM state, which performs batch processing
    /// using the underlying inference engine wrapper.
    /// - Parameters:
    ///   - prompts: An array of prompt strings to process in batch.
    ///   - completion: A closure invoked with the list of response strings.
    func getBatchLLMResponse(prompts: [String], completion: @escaping ([String]) -> Void) {
        Task { [weak self] in
            guard let self = self else { return }
            await self.llmState.processBatchTestContent(prompts, llm: self.llm) { responses in
                completion(responses)
            }
        }
    }

    func setModelConfig() {
        if let configStr = modelConfigManager.readConfigAsJSONString(), let llm = llm {
            llm.setConfigWithJSONString(configStr)
            llm.setVideoMaxFrames(modelConfigManager.readVideoMaxFrames())
        }
    }

    func updateVideoMaxFrames(_ value: Int) {
        modelConfigManager.saveVideoMaxFrames(value)
        llm?.setVideoMaxFrames(value)
    }

    func updateDefaultMultimodalPrompt(_ prompt: String) {
        modelConfigManager.saveDefaultMultimodalPrompt(prompt)
    }

    func updateEnableAudioOutput(_ enable: Bool) {
        print("[AudioViewModel] updateEnableAudioOutput: \(enable)")
        modelConfigManager.saveEnableAudioOutput(enable)
        llm?.setEnableAudioOutput(enable)
    }

    func updateTalkerSpeaker(_ speaker: String) {
        print("[AudioViewModel] updateTalkerSpeaker: \(speaker)")
        modelConfigManager.saveTalkerSpeaker(speaker)
        llm?.setTalkerSpeaker(speaker)
    }
    
    private func setupAudioOutput() {
        print("[AudioViewModel] setupAudioOutput called for model: \(modelInfo.name)")
        
        // Only setup audio for Omni models
        guard ModelUtils.supportAudioOutput(modelInfo.name) else {
            print("[AudioViewModel] Model does not support audio output, skipping setup")
            return
        }
        
        print("[AudioViewModel] Model supports audio output, initializing...")
        
        // Initialize audio playback manager
        if audioPlaybackManager == nil {
            print("[AudioViewModel] Creating AudioPlaybackManager")
            audioPlaybackManager = AudioPlaybackManager()
            audioPlaybackManager?.start()
        } else {
            print("[AudioViewModel] AudioPlaybackManager already exists")
        }
        
        // Configure audio output settings
        let enableAudio = modelConfigManager.readEnableAudioOutput()
        let talkerSpeaker = modelConfigManager.readTalkerSpeaker()
        
        print("[AudioViewModel] Configuring audio: enable=\(enableAudio), speaker=\(talkerSpeaker)")
        
        llm?.setEnableAudioOutput(enableAudio)
        llm?.setTalkerSpeaker(talkerSpeaker)
        
        // Set up audio waveform callback
        var audioChunkCount = 0
        var audioLastSeen = false
        print("[AudioViewModel] Setting up audio waveform callback")
        llm?.setAudioWaveformCallback { [weak self] data, size, isLastChunk in
            guard let self = self else {
                print("[AudioViewModel] Callback: self is nil, returning")
                return false
            }
            
            audioChunkCount += 1
            audioLastSeen = isLastChunk
            print("[AudioViewModel] chunk #\(audioChunkCount), size=\(size), isLastChunk=\(isLastChunk)")
            
            if isLastChunk {
                print("[AudioViewModel] tail received at #\(audioChunkCount)")
            }
            
            print("[AudioViewModel] Audio waveform callback: size=\(size), isLastChunk=\(isLastChunk)")
            
            // Convert C array to Swift array
            let floatArray = Array(UnsafeBufferPointer(start: data, count: Int(size)))
            
            // Check for NaN or invalid values and filter them
            let validArray = floatArray.map { value -> Float in
                if value.isNaN || value.isInfinite {
                    return 0.0
                }
                // Clamp to valid audio range [-1.0, 1.0]
                return max(-1.0, min(1.0, value))
            }
            
            // Check if we have any non-zero valid data
            let hasValidData = validArray.contains { abs($0) > 0.0001 }
            if !hasValidData && !isLastChunk {
                print("[AudioViewModel] Warning: Audio chunk contains only zeros/NaN, skipping playback (size=\(size))")
                // Don't skip if it's the last chunk, as it might be silence
                return false
            }
            
            // Log data statistics for debugging
            if size > 0 {
                let maxVal = validArray.max() ?? 0
                let minVal = validArray.min() ?? 0
                let avgVal = validArray.reduce(0, +) / Float(validArray.count)
                print("[AudioViewModel] Audio data stats: min=\(minVal), max=\(maxVal), avg=\(avgVal), hasValid=\(hasValidData)")
            }
            
            // Play audio chunk
            DispatchQueue.main.async {
                self.audioPlaybackManager?.playChunk(data: validArray, isLastChunk: isLastChunk)
            }
            
            // Return false to continue, true to stop
            return false
        }
        
        print("[AudioViewModel] Audio output setup completed")
    }

    // MARK: - Public Methods for File Operations

    /// Cleans the model temporary folder using FileOperationManager
    func cleanModelTmpFolder() {
        FileOperationManager.shared.cleanModelTempFolder(modelPath: modelInfo.localPath)
    }

    func updateUseMultimodalPromptAPI(_ value: Bool) {
        useMultimodalPromptAPI = value
        modelConfigManager.saveUseMultimodalPromptAPI(value)
    }

    /// Reloads the currently selected model to apply config changes that require recreation.
    func reloadCurrentModel() {
        guard !isProcessing else { return }
        contextUsageTimer?.cancel()
        contextUsageTimer = nil
        llm?.cancelInference()
        llm = nil
        diffusion = nil
        sanaDiffusion = nil
        setupLLM(modelPath: modelInfo.localPath)
    }

    func retryModelLoad() {
        guard loadFailed, !isProcessing else { return }
        reloadCurrentModel()
    }

    func onStart() {
        guard !isStarted else { return }
        isStarted = true
        interactor.messages
            .map { messages in
                messages.map { $0.toChatMessage() }
            }
            .sink { messages in
                self.messages = messages
            }
            .store(in: &subscriptions)

        interactor.connect()

        historySaveTimer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
            .sink { [weak self] _ in self?.saveHistoryCheckpoint() }

        setupLLM(modelPath: modelInfo.localPath)

        recordModelUsage()
    }

    func onStop() {
        guard isStarted else { return }
        isStarted = false
        modelLoadGeneration &+= 1
        contextGeneration &+= 1
        activeRequestGeneration = nil
        isRecognizingPreset = false
        historySaveTimer?.cancel()
        historySaveTimer = nil
        contextUsageTimer?.cancel()
        contextUsageTimer = nil
        contextSummary?.save(historyId: historyId)
        presetASRGeneration &+= 1
        presetASRService.cancel()
        recordModelUsage()

        saveHistoryImmediately()

        subscriptions.removeAll()

        interactor.disconnect()

        llm?.cancelInference()

        // Destroying the engine frees the model and can take a noticeable
        // time; do it off the main thread so switching models stays responsive.
        if let engine = llm {
            llm = nil
            DispatchQueue.global(qos: .userInitiated).async { withExtendedLifetime(engine) {} }
        }
        isModelLoaded = false
        isProcessing = false
        isStopping = false
        requestThinkingOverride = nil
        clearAllStreamingStates()
        diffusion = nil
        sanaDiffusion = nil

        FileOperationManager.shared.cleanTempDirectories()
        if !useMmap {
            FileOperationManager.shared.cleanModelTempFolder(modelPath: modelInfo.localPath)
        }
    }

    func loadMoreMessage(before _: Message) {
        interactor.loadNextPage()
            .sink { _ in }
            .store(in: &subscriptions)
    }

    private func recordModelUsage() {
        ModelStorageManager.shared.updateLastUsed(for: modelInfo.modelName)

        NotificationCenter.default.post(
            name: .modelUsageUpdated,
            object: nil,
            userInfo: ["modelName": modelInfo.modelName]
        )
    }

    /**
     * Called when streaming animation completes
     * Clears the currentStreamingMessageId to update UI state
     */
    @objc func onStreamingAnimationComplete(_ notification: Notification) {
        guard let messageId = notification.userInfo?["messageId"] as? String else {
            return
        }

        DispatchQueue.main.async { [weak self] in
            self?.finishStreamingPresentation(for: messageId)
        }
    }

    func finishStreamingPresentation(for messageId: String) {
        guard let stateManager = streamingStates[messageId] else { return }

        stateManager.markAnimationComplete()
        guard stateManager.state.isFullyComplete else { return }

        streamingStates.removeValue(forKey: messageId)
        if messageId == currentStreamingMessageId {
            currentStreamingMessageId = nil
        }
        if let performanceData = pendingPerformanceData.removeValue(forKey: messageId) {
            interactor.updatePerformanceData(performanceData, for: messageId)
        }
    }

    // MARK: - Streaming State Helpers

    /// Get the streaming state of a message
    func getStreamingState(_ messageId: String) -> StreamingMessageState? {
        return streamingStates[messageId]?.state
    }

    /// Get the streaming state of a message (convenience method with default value)
    func getStreamingState(for messageId: String) -> StreamingMessageState {
        return streamingStates[messageId]?.state ?? .none
    }

    /// Check if a message is in streaming state
    func isMessageStreaming(_ messageId: String) -> Bool {
        return streamingStates[messageId]?.state.isStreaming ?? false
    }

    /// Force complete streaming message (for error handling or cleanup)
    func forceCompleteStreaming(for messageId: String) {
        if let stateManager = streamingStates[messageId] {
            stateManager.forceComplete()
            streamingStates.removeValue(forKey: messageId)
            pendingPerformanceData.removeValue(forKey: messageId)
            if messageId == currentStreamingMessageId {
                currentStreamingMessageId = nil
                isProcessing = false
            }
        }
    }

    /// Clear all streaming states (for reset or error recovery)
    func clearAllStreamingStates() {
        streamingStates.removeAll()
        pendingPerformanceData.removeAll()
        currentStreamingMessageId = nil
    }
}
