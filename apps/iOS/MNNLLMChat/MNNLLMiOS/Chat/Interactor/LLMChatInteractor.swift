//
//  LLMChatInteractor.swift
//  MNNLLMiOS
//  Modified from ExyteChat's Chat Example
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

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
    
    private let processor = ThinkResultProcessor(thinkingPrefix: "<think>", completePrefix: "</think>")
    private lazy var chatState = CurrentValueSubject<[LLMChatMessage], Never>(generateStartMessages(historyMessages: historyMessages))
    private lazy var sharedState = chatState.share()

    private var isLoading = false
    private var lastDate = Date()

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
        self.chatData = LLMChatData(modelInfo: modelInfo)
        self.historyMessages = historyMessages
    }
    
    deinit {
        print("yxy:: LLMChatInteractor deinit")
    }

    func send(draftMessage: ExyteChat.DraftMessage, userType: UserType) {
        if draftMessage.id != nil {
            guard let index = chatState.value.firstIndex(where: { $0.uid == draftMessage.id }) else {
                return
            }
            chatState.value.remove(at: index)
        }
        
        Task {
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
                status: status)
            
            DispatchQueue.main.async { [weak self] in
                
//                PerformanceMonitor.shared.recordUIUpdate()
                
                switch userType {
                case .user, .system:
                    self?.chatState.value.append(message)
                    
                    if userType == .user {
                        sender = self?.chatData.assistant ?? message.sender
                        let emptyMessage = LLMChatMessage(uid: UUID().uuidString, sender: sender, createdAt: Date(), text: "", images: [], videos: [], recording: nil, replyMessage: nil)
                        self?.chatState.value.append(emptyMessage)
                    } else {
                        sender = self?.chatData.system ?? message.sender
                    }
                    
                    self?.processor.startNewChat()
                    
                case .assistant:
                    
//                    PerformanceMonitor.shared.measureExecutionTime(operation: "String concatenation") {
                        var updateLastMsg = self?.chatState.value[(self?.chatState.value.count ?? 1) - 1]
                        
                        if self?.modelInfo.tags.contains("Think") == true,
                            let text = self?.processor.process(progress: message.text) {
                            updateLastMsg?.text = text
                        } else {
                            if let currentText = updateLastMsg?.text {
                                updateLastMsg?.text = currentText + message.text
                            } else {
                                updateLastMsg?.text = message.text
                            }
                        }
                        
                        if let updatedMsg = updateLastMsg {
                            self?.chatState.value[(self?.chatState.value.count ?? 1) - 1] = updatedMsg
                        }
//                    }
                }
            }
        }
    }
    
    func sendImage(imageURL: URL) {
        Task {
            DispatchQueue.main.async { [weak self] in
                let image = LLMChatImage(id: UUID().uuidString, thumbnail: imageURL, full: imageURL)
                
                let message = LLMChatMessage.init(uid: UUID().uuidString, sender: self?.chatData.assistant ?? LLMChatUser(uid: "0", name: ""), createdAt: Date(), text: "", images: [image], videos: [], recording: nil, replyMessage: nil)
                self?.chatState.value.append(message)
            }
        }
    }

    func connect() {
        Timer.publish(every: 2, on: .main, in: .default)
            .autoconnect()
            .sink { [weak self] _ in
                self?.updateSendingStatuses()
            }
            .store(in: &subscriptions)
    }

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

    func updateSendingStatuses() {
        let updated = chatState.value.map {
            var message = $0
            if message.status == .sending {
                message.status = .sent
            } else if message.status == .sent {
                message.status = .read
            }
            return message
        }
        chatState.value = updated
    }
}
