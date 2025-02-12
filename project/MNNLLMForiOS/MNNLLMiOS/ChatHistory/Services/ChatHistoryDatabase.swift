//
//  ChatHistoryDatabase.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import SQLite
import ExyteChat

class ChatHistoryDatabase {
    static let shared: ChatHistoryDatabase? = {
        do {
            return try ChatHistoryDatabase()
        } catch {
            print("Failed to initialize ChatHistoryDatabase: \(error)")
            return nil
        }
    }()
    
    private let db: Connection
    private let chatHistories: Table
    
    private typealias Column<T> = SQLite.Expression<T>
    
    private let id: Column<String>
    private let modelId: Column<String>
    private let modelName: Column<String>
    private let messages: Column<String>
    private let createdAt: Column<Date>
    private let updatedAt: Column<Date>
    
    private init() throws {
        let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first!
        db = try Connection("\(path)/chatHistory.sqlite3")
        
        chatHistories = Table("chatHistories")
        id = Column<String>("id")
        modelId = Column<String>("modelId")
        modelName = Column<String>("modelName")
        messages = Column<String>("messages")
        createdAt = Column<Date>("createdAt")
        updatedAt = Column<Date>("updatedAt")
        
        try db.run(chatHistories.create(ifNotExists: true) { t in
            t.column(id, primaryKey: true)
            t.column(modelId)
            t.column(modelName)
            t.column(messages)
            t.column(createdAt)
            t.column(updatedAt)
        })
    }
    
    func saveChat(historyId: String, modelId: String, modelName: String, messages: [Message]) {
        do {
            ChatHistoryFileManager.shared.createHistoryDirectory(for: historyId)
            
            var historyMessages: [HistoryMessage] = []
            for message in messages {
                
                var copiedImages:[LLMChatImage] = []
                for msg in message.attachments {
                    if msg.type == .image {
                        
                        var imageUrl = msg.full
                        guard let copiedImage = ChatHistoryFileManager.shared.copyFile(from: imageUrl, for: historyId) else {
                            continue
                        }
                        
                        if copiedImage.isHEICImage() {
                            guard let jpgUrl = AssetExtractor.convertHEICToJPG(heicUrl: copiedImage) else { continue }
                            imageUrl = jpgUrl
                        }
                        
                        copiedImages.append(LLMChatImage.init(id: msg.id, thumbnail: imageUrl, full: imageUrl))
                    }
                }
                
                var copiedRecording: Recording?
                if let recording = message.recording, let recUrl = recording.url {
                    let recUrl = ChatHistoryFileManager.shared.copyFile(from: recUrl, for: historyId)
                    copiedRecording = Recording.init(duration: recording.duration, waveformSamples: recording.waveformSamples, url: recUrl)
                }
                
                historyMessages.append(HistoryMessage(
                    id: message.id,
                    content: message.text,
                    images: copiedImages,
                    audio: copiedRecording,
                    isUser: message.user.isCurrentUser,
                    createdAt: message.createdAt
                ))
            }
            
            let messagesData = try JSONEncoder().encode(historyMessages)
            let messagesString = String(data: messagesData, encoding: .utf8)!
            
            if let existingHistory = try? db.pluck(chatHistories.filter(id == historyId)) {
                try db.run(chatHistories.filter(id == historyId).update(
                    self.messages <- messagesString,
                    updatedAt <- Date()
                ))
            } else {
                try db.run(chatHistories.insert(
                    self.id <- historyId,
                    self.modelId <- modelId,
                    self.modelName <- modelName,
                    self.messages <- messagesString,
                    self.createdAt <- Date(),
                    self.updatedAt <- Date()
                ))
            }
        } catch {
            print("Failed to save chat: \(error)")
        }
    }
    
    func getAllHistory() -> [ChatHistory] {
        var histories: [ChatHistory] = []
        
        do {
            for history in try db.prepare(chatHistories) {
                let messagesData = history[messages].data(using: .utf8)!
                let historyMessages = try JSONDecoder().decode([HistoryMessage].self, from: messagesData)
                
                let chatHistory = ChatHistory(
                    id: history[id],
                    modelId: history[modelId],
                    modelName: history[modelName],
                    messages: historyMessages,
                    createdAt: history[createdAt],
                    updatedAt: history[updatedAt]
                )
                
                histories.append(chatHistory)
            }
        } catch {
            print("Failed to fetch histories: \(error)")
        }
        
        return histories
    }
    
    func deleteHistory(_ history: ChatHistory) {
        do {
            try db.run(chatHistories.filter(id == history.id).delete())
            ChatHistoryFileManager.shared.deleteHistoryDirectory(for: history.id)
        } catch {
            print("Failed to delete history: \(error)")
        }
    }
} 
