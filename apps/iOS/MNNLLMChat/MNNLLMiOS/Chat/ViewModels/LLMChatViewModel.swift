//
//  LLMChatViewModel.swift
//  MNNLLMiOS
//  Created by 游薪渝(揽清) on 2025/9/29.
//

import AVFoundation
import Combine
import SwiftUI
import UIKit

import ExyteChat

final class LLMChatViewModel: ObservableObject, StreamingMessageProvider {
    private var llm: LLMInferenceEngineWrapper?
    private var diffusion: DiffusionSession?
    private var sanaDiffusion: SanaDiffusionSession?
    private let llmState = LLMState()
    private var audioPlaybackManager: AudioPlaybackManager?

    @Published var messages: [Message] = []
    @Published var isModelLoaded = false
    @Published var isProcessing: Bool = false
    @Published var currentStreamingMessageId: String? = nil
    @Published var streamingStates: [String: StreamingMessageStateManager] = [:]

    @Published var useMmap: Bool = false
    @Published var useMultimodalPromptAPI: Bool = true

    // MARK: - Think Mode Properties

    @Published var isThinkingModeEnabled: Bool = true
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
        return modelInfo.modelName.lowercased().contains("stable-diffusion")
    }

    var isSanaDiffusionModel: Bool {
        return ModelUtils.isSanaDiffusionModel(modelInfo.modelName)
    }

    var isAnyDiffusionModel: Bool {
        return isDiffusionModel || isSanaDiffusionModel
    }

    init(modelInfo: ModelInfo, history: ChatHistory? = nil) {
        self.modelInfo = modelInfo
        self.history = history
        historyId = history?.id ?? UUID().uuidString
        let messages = self.history?.messages
        interactor = LLMChatInteractor(modelInfo: modelInfo, historyMessages: messages)

        modelConfigManager = ModelConfigManager(modelPath: modelInfo.localPath)

        useMmap = modelConfigManager.readUseMmap()
        useMultimodalPromptAPI = modelConfigManager.readUseMultimodalPromptAPI()

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
    func toggleThinkingMode() {
        guard supportsThinkingMode else { return }

        isThinkingModeEnabled.toggle()

        configureThinkingMode()

        print("Think mode toggled to: \(isThinkingModeEnabled)")
    }

    func setupLLM(modelPath: String) {
        Task { @MainActor in
            self.isModelLoaded = false
            do {
                try await self.send(draft: DraftMessage(
                    text: NSLocalizedString("ModelLoadingText", comment: ""),
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

        if isSanaDiffusionModel {
            // Load Sana Diffusion model for style transfer
            sanaDiffusion = SanaDiffusionSession(modelPath: modelPath, completion: { [weak self] success in
                Task { @MainActor in
                    print("Sana Diffusion Model loaded: \(success)")
                    self?.sendModelLoadStatus(success: success)
                    self?.isModelLoaded = success

                    // Set default prompt for Sana Diffusion
                    if success {
                        self?.defaultInputText = LLMChatViewModel.sanaDiffusionDefaultPrompt
                    }
                }
            })
        } else if isDiffusionModel {
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
                        self?.setModelConfig()
                        self?.configureThinkingMode()
                        self?.setupAudioOutput()
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

        Task {
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

    /// Sends a draft message to the LLM for processing
    /// - Parameter draft: The draft message to send
    func sendToLLM(draft: DraftMessage) {
        NotificationCenter.default.post(name: .dismissKeyboard, object: nil)

        Task {
            do {
                // Update Message UI and wait for completion
                try await send(draft: draft, userType: .user)

                recordModelUsage()

                if isModelLoaded {
                    if isSanaDiffusionModel {
                        getSanaDiffusionResponse(draft: draft)
                    } else if isDiffusionModel {
                        getDiffusionResponse(draft: draft)
                    } else {
                        getLLMRespsonse(draft: draft)
                    }
                }
            } catch {
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
                        text: NSLocalizedString("Please select an image for style transfer.", comment: ""),
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
                if let processedUrl = FileOperationManager.shared.processImageFile(from: url, fileName: fileName) {
                    inputImagePath = processedUrl.path
                    break
                }
            }

            guard let inputPath = inputImagePath else {
                try? await self.send(
                    draft: DraftMessage(
                        text: NSLocalizedString("Unsupported image format. Please use JPG/JPEG images for style transfer.", comment: ""),
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
            let initialStage = NSLocalizedString("Starting style transfer...", comment: "")
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
                            stageText = NSLocalizedString("Processing prompt...", comment: "Sana diffusion progress stage")
                        } else if progress >= 95 {
                            stageText = NSLocalizedString("Generating image...", comment: "Sana diffusion progress stage")
                        } else {
                            stageText = NSLocalizedString("Running diffusion...", comment: "Sana diffusion progress stage")
                        }
                        self.interactor.updateLastMessage(text: "\(stageText) (\(progress)%)")
                    }
                },
                completion: { [weak self] success, error, totalTimeMs in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.isProcessing = false

                        if success {
                            let completionText = NSLocalizedString("Style transfer completed!", comment: "")
                            self.interactor.updateLastMessage(text: completionText)
                            self.interactor.sendImage(imageURL: URL(fileURLWithPath: outputPath))

                            // Send total time as a separate message after the image
                            let totalTimeSec = totalTimeMs / 1000.0
                            let timeText = String(format: "%.1f", totalTimeSec)
                            let timeMessage = NSLocalizedString("Total time:", comment: "Sana diffusion total time label") + " \(timeText)s"
                            do {
                                try await self.send(draft: DraftMessage(text: timeMessage, thinkText: "", useMarkdown: false, medias: [], recording: nil, replyMessage: nil, createdAt: Date()), userType: .system)
                            } catch {
                                print("Error sending time message: \(error)")
                            }
                            
                        } else {
                            let errorMessage = error ?? NSLocalizedString("Style transfer failed.", comment: "")
                            self.interactor.updateLastMessage(text: errorMessage)
                        }
                    }
                }
            )
        }
    }

    func getLLMRespsonse(draft: DraftMessage) {
        Task {
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
                    let fileName = url.lastPathComponent

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
                let warningText = NSLocalizedString(
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

            // First, send the empty message asynchronously
            let emptyMessage = DraftMessage(
                text: "",
                thinkText: "",
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

            let convertedContent = self.convertDeepSeekMutliChat(content: content)

            let outputHandler: (String) -> Void = { [weak self] output in
                guard let self = self else { return }

                if output.contains("<eop>") {
                    Task {
                        await UIUpdateOptimizer.shared.forceFlush { [weak self] finalOutput in
                            guard let self = self else { return }
                            if !finalOutput.isEmpty {
                                Task {
                                    do {
                                        try await self.send(draft: DraftMessage(
                                            text: finalOutput,
                                            thinkText: "",
                                            medias: [],
                                            recording: nil,
                                            replyMessage: nil,
                                            createdAt: Date()
                                        ), userType: .assistant)
                                    } catch {
                                        print("Error sending final output message: \(error)")
                                    }
                                }
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
                        Task {
                            do {
                                try await self.send(draft: DraftMessage(
                                    text: output,
                                    thinkText: "",
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
                }
            }

            if shouldUseMultimodalAPI {
                await llmState.processMultimodalContent(
                    convertedContent,
                    images: imageDictionary,
                    llm: self.llm,
                    showPerformance: true,
                    completion: outputHandler
                )
            } else {
                await llmState.processContent(
                    convertedContent,
                    llm: self.llm,
                    showPerformance: true,
                    completion: outputHandler
                )
            }
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
        print("[AudioViewModel] setupAudioOutput called for model: \(modelInfo.modelName)")
        
        // Only setup audio for Omni models
        guard ModelUtils.supportAudioOutput(modelInfo.modelName) else {
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

    private func convertDeepSeekMutliChat(content: String) -> String {
        if modelInfo.modelName.lowercased().contains("deepseek") {
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

    func updateUseMultimodalPromptAPI(_ value: Bool) {
        useMultimodalPromptAPI = value
        modelConfigManager.saveUseMultimodalPromptAPI(value)
    }

    /// Reloads the currently selected model to apply config changes that require recreation.
    func reloadCurrentModel() {
        llm?.cancelInference()
        llm = nil
        setupLLM(modelPath: modelInfo.localPath)
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
