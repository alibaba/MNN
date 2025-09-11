//
//  LLMChatViewModel.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/1/8.
//

import AVFoundation
import Combine
import SwiftUI

import ExyteChat

final class LLMChatViewModel: ObservableObject, StreamingMessageProvider {
    private var llm: LLMInferenceEngineWrapper?
    private var diffusion: DiffusionSession?
    private let llmState = LLMState()

    @Published var messages: [Message] = []
    @Published var isModelLoaded = false
    @Published var isProcessing: Bool = false
    @Published var currentStreamingMessageId: String? = nil
    @Published var streamingStates: [String: StreamingMessageStateManager] = [:]

    @Published var useMmap: Bool = false

    // MARK: - Think Mode Properties

    @Published var isThinkingModeEnabled: Bool = true
    @Published var supportsThinkingMode: Bool = false

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
            "Model Loading..."
        }
    }

    var chatCover: URL? {
        interactor.otherSenders.count == 1 ? interactor.otherSenders.first?.avatar : nil
    }

    private let interactor: LLMChatInteractor
    private var subscriptions = Set<AnyCancellable>()

    var modelInfo: ModelInfo
    var history: ChatHistory?
    private var historyId: String

    let modelConfigManager: ModelConfigManager

    var isDiffusionModel: Bool {
        return modelInfo.modelName.lowercased().contains("diffusion")
    }

    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        // print("yxy:: LLMChat View Model init")
        self.modelInfo = modelInfo
        self.history = history
        historyId = history?.id ?? UUID().uuidString
        let messages = self.history?.messages
        interactor = LLMChatInteractor(modelInfo: modelInfo, historyMessages: messages)

        modelConfigManager = ModelConfigManager(modelPath: modelInfo.localPath)

        useMmap = modelConfigManager.readUseMmap()

        // Check if model supports thinking mode
        supportsThinkingMode = ModelUtils.isSupportThinkingSwitch(modelInfo.tags, modelName: modelInfo.modelName)

        // Listen for streaming animation completion notifications
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(onStreamingAnimationComplete(_:)),
            name: NSNotification.Name("StreamingAnimationCompleted"),
            object: nil
        )
    }

    deinit {
        // print("yxy:: LLMChat View Model deinit")
        // Cancel ongoing inference
        llm?.cancelInference()
        llm = nil
        isProcessing = false
        diffusion = nil

        // Clean up streaming states
        clearAllStreamingStates()

        // Remove notification observers
        NotificationCenter.default.removeObserver(self)

        print("yxy:: LLMChat View Model cleanup complete")
    }

    // MARK: - Think Mode Methods

    /// Toggle thinking mode on/off
    func toggleThinkingMode() {
        guard supportsThinkingMode else { return }

        isThinkingModeEnabled.toggle()

        configureThinkingMode()

        print("Think mode toggled to: \(isThinkingModeEnabled)")
    }

    func setupLLM(modelPath: String) {
        Task { @MainActor in
            self.isModelLoaded = false
            self.send(draft: DraftMessage(
                text: NSLocalizedString("ModelLoadingText", comment: ""),
                thinkText: "",
                medias: [],
                recording: nil,
                replyMessage: nil,
                createdAt: Date()
            ), userType: .system)
        }

        if modelInfo.modelName.lowercased().contains("diffusion") {
            diffusion = DiffusionSession(modelPath: modelPath, completion: { [weak self] success in
                Task { @MainActor in
                    print("Diffusion Model \(success)")
                    self?.sendModelLoadStatus(success: success)
                    self?.isModelLoaded = success
                }
            })
        } else {
            llm = LLMInferenceEngineWrapper(modelPath: modelPath) { [weak self] success in
                Task { @MainActor in
                    self?.sendModelLoadStatus(success: success)
                    self?.processHistoryMessages()
                    self?.isModelLoaded = success

                    // Configure thinking mode after model is loaded
                    if success {
                        self?.configureThinkingMode()
                    }
                }
            }
        }
    }

    /// Configure thinking mode after model loading
    private func configureThinkingMode() {
        guard let llm = llm, supportsThinkingMode else { return }

        if supportsThinkingMode {
            llm.setThinkingModeEnabled(isThinkingModeEnabled)
        }

        interactor.isThinkingModeEnabled = isThinkingModeEnabled

        print("Thinking mode configured: \(isThinkingModeEnabled)")
    }

    private func sendModelLoadStatus(success: Bool) {
        let modelLoadSuccessText = NSLocalizedString("ModelLoadingSuccessText", comment: "")
        let modelLoadFailText = NSLocalizedString("ModelLoadingFailText", comment: "")
        let loadResult = success ? modelLoadSuccessText : modelLoadFailText

        send(draft: DraftMessage(
            text: loadResult,
            thinkText: "",
            medias: [],
            recording: nil,
            replyMessage: nil,
            createdAt: Date()
        ), userType: .system)
    }

    private func processHistoryMessages() {
        guard let history = history else { return }

        let historyPrompts = history.messages.flatMap { msg -> [[String: String]] in
            var prompts: [[String: String]] = []
            let sender = msg.isUser ? "user" : "assistant"

            prompts.append([sender: msg.content])

            if let images = msg.images {
                let imgStr = images.map { "<img>\($0.full.path)</img>" }.joined()
                prompts.append([sender: imgStr])
            }

            if let audio = msg.audio, let url = audio.url {
                prompts.append([sender: "<audio>\(url.path)</audio>"])
            }

            return prompts
        }

        let nsArray = historyPrompts as [[AnyHashable: Any]]
        llm?.addPrompts(from: nsArray)
    }

    func sendToLLM(draft: DraftMessage) {
        NotificationCenter.default.post(name: .dismissKeyboard, object: nil)

        send(draft: draft, userType: .user)

        recordModelUsage()

        if isModelLoaded {
            if modelInfo.modelName.lowercased().contains("diffusion") {
                getDiffusionResponse(draft: draft)
            } else {
                getLLMRespsonse(draft: draft)
            }
        }
    }

    func send(draft: DraftMessage, userType: UserType) {
        interactor.send(draftMessage: draft, userType: userType)
    }

    func getDiffusionResponse(draft: DraftMessage) {
        Task {
            let tempImagePath = FileOperationManager.shared.generateTempImagePath().path

            var lastProcess: Int32 = 0

            self.send(draft: DraftMessage(text: "Start Generating Image...", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .assistant)

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
                                   self.send(draft: DraftMessage(text: "Image generated successfully!", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                                   self.interactor.sendImage(imageURL: URL(fileURLWithPath: tempImagePath))
                               } else if (progress - lastProcess) > 20 {
                                   lastProcess = progress
                                   self.send(draft: DraftMessage(text: "Generating Image \(progress)%", thinkText: "", medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                               }
                           })
        }
    }

    func getLLMRespsonse(draft: DraftMessage) {
        Task {
            await llmState.setProcessing(true)
            await MainActor.run {
                self.isProcessing = true
                let emptyMessage = DraftMessage(
                    text: "",
                    thinkText: "",
                    medias: [],
                    recording: nil,
                    replyMessage: nil,
                    createdAt: Date()
                )
                self.send(draft: emptyMessage, userType: .assistant)
                if let lastMessage = self.messages.last {
                    self.currentStreamingMessageId = lastMessage.id

                    // Create and start state manager
                    let stateManager = StreamingMessageStateManager(messageId: lastMessage.id)
                    self.streamingStates[lastMessage.id] = stateManager
                    stateManager.startStreaming()
                }
            }

            var content = draft.text
            let medias = draft.medias

            // MARK: Add image

            for media in medias {
                guard media.type == .image, let url = await media.getURL() else {
                    continue
                }

                let fileName = url.lastPathComponent

                if let processedUrl = FileOperationManager.shared.processImageFile(from: url, fileName: fileName) {
                    content = "<img>\(processedUrl.path)</img>" + content
                }
            }

            if let audio = draft.recording, let path = audio.url {
                //                if let wavFile = await convertACCToWAV(accFileUrl: path) {
                content = "<audio>\(path.path)</audio>" + content
                //                }
            }

            let convertedContent = self.convertDeepSeekMutliChat(content: content)

            await llmState.processContent(convertedContent, llm: self.llm, showPerformance: true) { [weak self] output in
                guard let self = self else { return }

                if output.contains("<eop>") {
                    Task {
                        await UIUpdateOptimizer.shared.forceFlush { [weak self] finalOutput in
                            guard let self = self else { return }
                            if !finalOutput.isEmpty {
                                self.send(draft: DraftMessage(
                                    text: finalOutput,
                                    thinkText: "",
                                    medias: [],
                                    recording: nil,
                                    replyMessage: nil,
                                    createdAt: Date()
                                ), userType: .assistant)
                            }
                        }

                        await MainActor.run {
                            // Mark model output as complete
                            if let messageId = self.currentStreamingMessageId,
                               let stateManager = self.streamingStates[messageId]
                            {
                                stateManager.markOutputComplete()
                            }
                            // currentStreamingMessageId will be cleared when animation completes via callback

                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                NotificationCenter.default.post(name: .dismissKeyboard, object: nil)
                            }
                        }
                        await self.llmState.setProcessing(false)
                    }
                    return
                }

                Task {
                    await UIUpdateOptimizer.shared.addUpdate(output) { [weak self] output in
                        guard let self = self else { return }
                        self.send(draft: DraftMessage(
                            text: output,
                            thinkText: "",
                            medias: [],
                            recording: nil,
                            replyMessage: nil,
                            createdAt: Date()
                        ), userType: .assistant)
                    }
                }
            }
        }
    }

    func setModelConfig() {
        if let configStr = modelConfigManager.readConfigAsJSONString(), let llm = llm {
            llm.setConfigWithJSONString(configStr)
        }
    }

    private func convertDeepSeekMutliChat(content: String) -> String {
        if modelInfo.modelName.lowercased().contains("deepseek") {
            /* formate:: <|begin_of_sentence|><|User|>{text}<|Assistant|>{text}<|end_of_sentence|>
             <|User|>{text}<|Assistant|>{text}<|end_of_sentence|>
             */
            var deepSeekContent = "<|begin_of_sentence|>"

            for message in messages {
                let senderTag: String
                switch message.user.id {
                case "1":
                    senderTag = "<|User|>"
                case "2":
                    senderTag = "<|Assistant|>"
                default:
                    continue
                }
                deepSeekContent += "\(senderTag)\(message.text)"
            }

            deepSeekContent += "<|end_of_sentence|><think><\n>"
            print(deepSeekContent)
            return deepSeekContent
        } else {
            return content
        }
    }

    // MARK: - Public Methods for File Operations

    /// Cleans the model temporary folder using FileOperationManager
    func cleanModelTmpFolder() {
        FileOperationManager.shared.cleanModelTempFolder(modelPath: modelInfo.localPath)
    }

    func onStart() {
        interactor.messages
            .map { messages in
                messages.map { $0.toChatMessage() }
            }
            .sink { messages in
                self.messages = messages
            }
            .store(in: &subscriptions)

        interactor.connect()

        setupLLM(modelPath: modelInfo.localPath)

        recordModelUsage()
    }

    func onStop() {
        recordModelUsage()

        ChatHistoryManager.shared.saveChat(
            historyId: historyId,
            modelInfo: modelInfo,
            messages: messages
        )

        subscriptions.removeAll()

        interactor.disconnect()

        llm?.cancelInference()

        llm = nil

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
        guard let messageId = notification.userInfo?["messageId"] as? String,
              let stateManager = streamingStates[messageId]
        else {
            return
        }

        DispatchQueue.main.async {
            // Mark animation as complete
            stateManager.markAnimationComplete()

            // Clean up state if fully complete
            if stateManager.state.isFullyComplete {
                self.streamingStates.removeValue(forKey: messageId)
                if messageId == self.currentStreamingMessageId {
                    self.isProcessing = false // MARK: isProcessing

                    self.currentStreamingMessageId = nil
                }
            }
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
            if messageId == currentStreamingMessageId {
                currentStreamingMessageId = nil
            }
        }
    }

    /// Clear all streaming states (for reset or error recovery)
    func clearAllStreamingStates() {
        streamingStates.removeAll()
        currentStreamingMessageId = nil
    }
}
