//
//  LLMChatInteractor.swift
//  MNNLLMiOS
//  Modified from ExyteChat's Chat Example
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import Foundation
import Combine
import ExyteChat

enum UserType {
    case system
    case user
    case assistant
}

final class LLMChatInteractor: ChatInteractorProtocol {
    var chatData: LLMChatData
    var modelInfo: ModelInfo
    var historyMessages: [HistoryMessage]?
    var isThinkingModeEnabled: Bool = true

    private let processor = ThinkResultProcessor()

    /// Qwen3.5-style models pre-open the think block in the chat template, so
    /// the stream carries no `<think>` opener — only `</think>`. Detect that
    /// from the model's config.json (`jinja.context.enable_thinking`).
    lazy var thinkingExpectedByConfig: Bool = {
        guard let data = FileManager.default.contents(atPath: modelInfo.localPath + "/config.json"),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let jinja = json["jinja"] as? [String: Any],
              let context = jinja["context"] as? [String: Any],
              let enabled = context["enable_thinking"] as? Bool else {
            return false
        }
        return enabled
    }()

    private lazy var chatState = CurrentValueSubject<[LLMChatMessage], Never>(generateStartMessages(historyMessages: historyMessages))
    private lazy var sharedState = chatState.share()

    private var isLoading = false
    private var lastDate = Date()
    #if DEBUG
    private var lastLoggedThinkLen = 0
    #endif

    private var subscriptions = Set<AnyCancellable>()

    var messages: AnyPublisher<[LLMChatMessage], Never> {
        sharedState.eraseToAnyPublisher()
    }

    var senders: [LLMChatUser] {
        let members = [chatData.assistant, chatData.user]
        return members
    }

    var otherSenders: [LLMChatUser] {
        senders.filter { !$0.isCurrentUser }
    }

    init(modelInfo: ModelInfo, historyMessages: [HistoryMessage]? = nil) {
        self.modelInfo = modelInfo
        chatData = LLMChatData(modelInfo: modelInfo)
        self.historyMessages = historyMessages
    }

    deinit {
    }

    /// Sends a draft message to the chat with the specified user type
    /// - Parameters:
    ///   - draftMessage: The draft message to send
    ///   - userType: The type of user sending the message (user, assistant, or system)
    /// - Throws: Any error that occurs during message processing
    func send(draftMessage: ExyteChat.DraftMessage, userType: UserType) async throws {
        if draftMessage.id != nil {
            guard let index = chatState.value.firstIndex(where: { $0.uid == draftMessage.id }) else {
                return
            }
            chatState.value.remove(at: index)
        }

        let status: Message.Status = .sending

        var sender: LLMChatUser
        switch userType {
        case .user:
            sender = chatData.user
        case .assistant:
            sender = chatData.assistant
        case .system:
            sender = chatData.system
        }

        let message: LLMChatMessage = await draftMessage.toLLMChatMessage(
            id: UUID().uuidString,
            user: sender,
            status: status
        )

        await MainActor.run { [weak self] in
            switch userType {
            case .user, .system:
                self?.chatState.value.append(message)

                if userType == .user {
                    let assistantSender = self?.chatData.assistant ?? message.sender
                    let emptyMessage = LLMChatMessage(
                        uid: UUID().uuidString,
                        sender: assistantSender,
                        createdAt: Date(),
                        useMarkdown: true,
                        text: "",
                        images: [],
                        videos: [],
                        recording: nil,
                        replyMessage: nil
                    )
                    self?.chatState.value.append(emptyMessage)
                }

                self?.processor.startNewChat()

            case .assistant:
                var updateLastMsg = self?.chatState.value[(self?.chatState.value.count ?? 1) - 1]
                updateLastMsg?.useMarkdown = message.useMarkdown

                if let performanceData = message.performanceData, !performanceData.isEmpty {
                    // Metrics are UI metadata, not model-authored Markdown.
                    updateLastMsg?.performanceData = performanceData
                } else if !message.text.isEmpty {
                    let tagsSayThinking = self?.modelInfo.tags.contains(where: {
                        $0.localizedCaseInsensitiveContains("Think") || $0.localizedCaseInsensitiveContains("思考")
                    }) ?? false
                    let thinkingApplies = (self?.isThinkingModeEnabled == true) &&
                        (tagsSayThinking || self?.thinkingExpectedByConfig == true)

                    #if DEBUG
                    // One-shot per round. NSLog with a pre-composed literal is
                    // visible in idevicesyslog (unlike %@ args or stderr here).
                    if self?.processor.answer.isEmpty == true && self?.processor.thinking.isEmpty == true {
                        NSLog("[ThinkSplit] gate enabled=%d tags=%d config=%d applies=%d",
                              self?.isThinkingModeEnabled == true ? 1 : 0,
                              tagsSayThinking ? 1 : 0,
                              self?.thinkingExpectedByConfig == true ? 1 : 0,
                              thinkingApplies ? 1 : 0)
                    }
                    #endif

                    if thinkingApplies, let processor = self?.processor {
                        processor.expectThinking = self?.thinkingExpectedByConfig == true
                        processor.process(delta: message.text)
                        // Split stream: thinking scratchpad goes to the folding
                        // sub-panel, the answer body to the main bubble.
                        updateLastMsg?.thinkText = processor.thinking
                        updateLastMsg?.text = processor.answer
                        #if DEBUG
                        // UTF-8 byte counts: O(1), unlike Character counts.
                        let thinkLen = processor.thinking.utf8.count
                        if thinkLen > 0, thinkLen % 1500 < 120, thinkLen != self?.lastLoggedThinkLen {
                            self?.lastLoggedThinkLen = thinkLen
                            NSLog("[ThinkSplit] thinkingBytes=\(thinkLen) answerBytes=\(processor.answer.utf8.count)")
                        }
                        #endif
                    } else {
                        if let currentText = updateLastMsg?.text {
                            updateLastMsg?.text = currentText + message.text
                        } else {
                            updateLastMsg?.text = message.text
                        }
                    }
                }

                if let updatedMsg = updateLastMsg {
                    self?.chatState.value[(self?.chatState.value.count ?? 1) - 1] = updatedMsg
                }
            }
        }
    }

    /// Appends a preset user message (optionally with an image) plus the empty assistant bubble,
    /// mirroring the .user branch of send(draftMessage:userType:) without requiring a DraftMessage.
    func sendPresetMessage(text: String, imageURL: URL?) async {
        let images: [LLMChatImage] = imageURL.map { url in
            [LLMChatImage(id: UUID().uuidString, thumbnail: url, full: url)]
        } ?? []
        let message = LLMChatMessage(
            uid: UUID().uuidString,
            sender: chatData.user,
            createdAt: Date(),
            status: .sending,
            text: text,
            images: images,
            videos: [],
            recording: nil,
            replyMessage: nil
        )

        await MainActor.run { [weak self] in
            self?.chatState.value.append(message)

            let assistantSender = self?.chatData.assistant ?? message.sender
            let emptyMessage = LLMChatMessage(
                uid: UUID().uuidString,
                sender: assistantSender,
                createdAt: Date(),
                useMarkdown: true,
                text: "",
                images: [],
                videos: [],
                recording: nil,
                replyMessage: nil
            )
            self?.chatState.value.append(emptyMessage)

            self?.processor.startNewChat()
        }
    }

    /// Appends a prerecorded audio preset with its ASR transcript and metrics.
    /// Audio and recognition evidence share one user bubble before LLM output.
    func sendPresetAudioMessage(text: String, recording: Recording, expectsResponse: Bool) async {
        let message = LLMChatMessage(
            uid: UUID().uuidString,
            sender: chatData.user,
            createdAt: Date(),
            status: .sending,
            useMarkdown: false,
            text: text,
            images: [],
            videos: [],
            recording: recording,
            replyMessage: nil
        )

        await MainActor.run { [weak self] in
            self?.chatState.value.append(message)
            if expectsResponse {
                let assistantSender = self?.chatData.assistant ?? message.sender
                let emptyMessage = LLMChatMessage(
                    uid: UUID().uuidString,
                    sender: assistantSender,
                    createdAt: Date(),
                    text: "",
                    images: [],
                    videos: [],
                    recording: nil,
                    replyMessage: nil
                )
                self?.chatState.value.append(emptyMessage)
                self?.processor.startNewChat()
            }
        }
    }

    func sendImage(imageURL: URL) {
        Task {
            await MainActor.run { [weak self] in
                let image = LLMChatImage(id: UUID().uuidString, thumbnail: imageURL, full: imageURL)

                let message = LLMChatMessage(uid: UUID().uuidString, sender: self?.chatData.assistant ?? LLMChatUser(uid: "0", name: ""), createdAt: Date(), text: "", images: [image], videos: [], recording: nil, replyMessage: nil)
                self?.chatState.value.append(message)
            }
        }
    }

    /// Replaces the text of the last message in the chat (e.g. for in-place progress updates).
    /// Uses DispatchQueue.main for immediate, synchronous-like updates without Task scheduling delay.
    func updateLastMessage(text: String) {
        let update = { [weak self] in
            guard let self = self, !self.chatState.value.isEmpty else { return }
            let lastIndex = self.chatState.value.count - 1
            self.chatState.value[lastIndex].text = text
        }
        if Thread.isMainThread {
            update()
        } else {
            DispatchQueue.main.async(execute: update)
        }
    }

    /// Empties the conversation (new-chat action for the Pocket app).
    func clearMessages() {
        let update = { [weak self] in
            guard let self else { return }
            self.chatState.value = []
        }
        if Thread.isMainThread {
            update()
        } else {
            DispatchQueue.main.async(execute: update)
        }
    }

    /// Replace the latest response, keeping its user message and earlier turns.
    @MainActor
    /// Drops everything after the user message (later turns included) and
    /// appends an empty answer to stream into.
    func beginRegeneration(after userId: String) -> Bool {
        guard let index = chatState.value.lastIndex(where: { $0.uid == userId && $0.sender.uid == "1" }) else { return false }
        var updated = Array(chatState.value[...index])
        updated.append(LLMChatMessage(uid: UUID().uuidString, sender: chatData.assistant,
                                     createdAt: Date(), text: "", images: [], videos: [],
                                     recording: nil, replyMessage: nil))
        processor.startNewChat()
        chatState.value = updated
        return true
    }

    /// Replaces the whole conversation, e.g. when another saved chat is opened
    /// with the model that is already loaded.
    func replaceMessages(with historyMessages: [HistoryMessage]?) {
        processor.startNewChat()
        chatState.value = generateStartMessages(historyMessages: historyMessages)
    }

    /// Sets a message's answer and thinking text directly (agent runs build
    /// both themselves instead of streaming through the think splitter).
    func setMessage(_ messageId: String, text: String, thinkText: String, toolSteps: [ToolStep]) {
        guard let index = chatState.value.firstIndex(where: { $0.uid == messageId }) else { return }
        var message = chatState.value[index]
        message.text = text
        message.thinkText = thinkText
        message.toolSteps = toolSteps
        message.useMarkdown = true
        chatState.value[index] = message
    }

    /// Removes the message and everything after it.
    func truncate(from messageId: String) -> Bool {
        guard let index = chatState.value.firstIndex(where: { $0.uid == messageId }) else { return false }
        processor.startNewChat()
        chatState.value = Array(chatState.value[..<index])
        return true
    }

    /// Read-only performance metadata for a message (demo answer view).
    func performanceData(for messageId: String) -> String? {
        chatState.value.first { $0.uid == messageId }?.performanceData
    }

    /// Attaches final performance metadata to its assistant message without
    /// mixing it into the model-authored Markdown content.
    func updatePerformanceData(_ performanceData: String, for messageId: String) {
        let update = { [weak self] in
            guard let self,
                  let index = self.chatState.value.firstIndex(where: { $0.uid == messageId })
            else {
                return
            }
            self.chatState.value[index].performanceData = performanceData
        }
        if Thread.isMainThread {
            update()
        } else {
            DispatchQueue.main.async(execute: update)
        }
    }

    func connect() {}

    func disconnect() {
        subscriptions.removeAll()
    }

    func loadNextPage() -> Future<Bool, Never> {
        Future<Bool, Never> { [weak self] promise in
            guard let self = self, !self.isLoading else {
                promise(.success(false))
                return
            }
            self.isLoading = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                guard let self = self else { return }
                let messages = self.generateStartMessages()
                self.chatState.value = messages + self.chatState.value
                self.isLoading = false
                promise(.success(true))
            }
        }
    }
}

private extension LLMChatInteractor {
    func generateStartMessages(historyMessages: [HistoryMessage]? = nil) -> [LLMChatMessage] {
        return chatData.greatingMessage(historyMessages: historyMessages)
    }
}
