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
    private let modelInfo: Column<String?> // JSON string of ModelInfo
    private let messages: Column<String>
    private let createdAt: Column<Date>
    private let updatedAt: Column<Date>
    
    private init() throws {
        guard let path = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first else {
            throw NSError(domain: "ChatHistoryDatabase", code: -1, userInfo: [NSLocalizedDescriptionKey: "Documents directory not found"])
        }
        db = try Connection("\(path)/chatHistory.sqlite3")
        
        chatHistories = Table("chatHistories")
        id = Column<String>("id")
        modelId = Column<String>("modelId")
        modelName = Column<String>("modelName")
        modelInfo = Column<String?>("modelInfo")
        messages = Column<String>("messages")
        createdAt = Column<Date>("createdAt")
        updatedAt = Column<Date>("updatedAt")
        
        try db.run(chatHistories.create(ifNotExists: true) { t in
            t.column(id, primaryKey: true)
            t.column(modelId)
            t.column(modelName)
            t.column(modelInfo)
            t.column(messages)
            t.column(createdAt)
            t.column(updatedAt)
        })
        
        // for old data
        try migrateDatabase()
    }
    
    private func migrateDatabase() throws {
        do {
            let _ = try db.scalar("SELECT modelInfo FROM chatHistories LIMIT 1")
        } catch {
            try db.run("ALTER TABLE chatHistories ADD COLUMN modelInfo TEXT")
        }
    }
    
    func saveChat(historyId: String, modelInfo: ModelInfo, messages: [Message]) {
        let modelId = modelInfo.id
        let modelName = modelInfo.modelName
        do {
            ChatHistoryFileManager.shared.createHistoryDirectory(for: historyId)
            
            var historyMessages: [HistoryMessage] = []
            for message in messages {
                
                var copiedImages:[LLMChatImage] = []
                for msg in message.attachments {
                    if msg.type == .image {
                        var imageUrl = msg.full
                        
                        print("Processing image attachment: \(imageUrl.path)")

                        guard let copiedImage = ChatHistoryFileManager.shared.copyFile(from: imageUrl, for: historyId) else {
                            print("Failed to copy image file: \(imageUrl.path)")
                            continue
                        }
                        
                        imageUrl = copiedImage
                        print("Image copied to: \(imageUrl.path)")
                        
                        if imageUrl.isHEICImage() {
                            guard let jpgUrl = AssetExtractor.convertHEICToJPG(heicUrl: imageUrl) else { 
                                print("Failed to convert HEIC to JPG: \(imageUrl.path)")
                                continue 
                            }
                            imageUrl = jpgUrl
                            print("HEIC converted to JPG: \(imageUrl.path)")
                        }
                        
                        if ChatHistoryFileManager.shared.validateFileExists(at: imageUrl) {
                            copiedImages.append(LLMChatImage.init(id: msg.id, thumbnail: imageUrl, full: imageUrl))
                            print("Image successfully saved for history: \(imageUrl.path)")
                        } else {
                            print("Final image file validation failed: \(imageUrl.path)")
                        }
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
            
            let modelInfoData = try JSONEncoder().encode(modelInfo)
            let modelInfoString = String(data: modelInfoData, encoding: .utf8)!
            
            if let _ = try? db.pluck(chatHistories.filter(id == historyId)) {
                try db.run(chatHistories.filter(id == historyId).update(
                    self.messages <- messagesString,
                    self.modelInfo <- modelInfoString,
                    updatedAt <- Date()
                ))
            } else {
                try db.run(chatHistories.insert(
                    self.id <- historyId,
                    self.modelId <- modelId,
                    self.modelName <- modelName,
                    self.modelInfo <- modelInfoString,
                    self.messages <- messagesString,
                    self.createdAt <- Date(),
                    self.updatedAt <- Date()
                ))
            }
        } catch {
            print("Failed to save chat: \(error)")
        }
    }
    
    // For backward compatibility
    func saveChat(historyId: String, modelId: String, modelName: String, messages: [Message]) {
        let modelInfo = ModelInfo(modelId: modelId, isDownloaded: true)
        saveChat(historyId: historyId, modelInfo: modelInfo, messages: messages)
    }
    
    func getAllHistory() -> [ChatHistory] {
        var histories: [ChatHistory] = []
        
        do {
            for history in try db.prepare(chatHistories) {
                let messagesData = history[messages].data(using: .utf8)!
                var historyMessages = try JSONDecoder().decode([HistoryMessage].self, from: messagesData)
                
                historyMessages = validateAndFixImagePaths(historyMessages, historyId: history[id])
                
                var modelInfoObj: ModelInfo
                do {
                    if let modelInfoString = try? history.get(modelInfo), !modelInfoString.isEmpty {
                        do {
                            let modelInfoData = modelInfoString.data(using: .utf8)!
                            modelInfoObj = try JSONDecoder().decode(ModelInfo.self, from: modelInfoData)
                            // print("Successfully decoded ModelInfo from JSON for history: \(history[id])")
                        } catch {
                            // print("Failed to decode ModelInfo from JSON, using fallback: \(error)")
                            modelInfoObj = ModelInfo(modelId: history[modelId], isDownloaded: true)
                        }
                    } else {
                        // For backward compatibility
                        // print("No modelInfo data found, using fallback for history: \(history[id])")
                        modelInfoObj = ModelInfo(modelId: history[modelId], isDownloaded: true)
                    }
                } catch {
                    // For backward compatibility
                    // print("ModelInfo column not found, using fallback for history: \(history[id])")
                    modelInfoObj = ModelInfo(modelId: history[modelId], isDownloaded: true)
                }
                
                let chatHistory = ChatHistory(
                    id: history[id],
                    modelInfo: modelInfoObj,
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
    
    private func validateAndFixImagePaths(_ messages: [HistoryMessage], historyId: String) -> [HistoryMessage] {
        return messages.map { message in
            var updatedMessage = message
            
            if let images = message.images {
                let validImages = images.compactMap { image -> LLMChatImage? in
                    if ChatHistoryFileManager.shared.validateFileExists(at: image.full) {
                        return image
                    } else {
                        let fileName = image.full.lastPathComponent
                        if let validURL = ChatHistoryFileManager.shared.getValidFileURL(for: fileName, historyId: historyId) {
                            return LLMChatImage(id: image.id, thumbnail: validURL, full: validURL)
                        } else {
                            print("Image file not found and cannot be recovered: \(image.full.path)")
                            return nil
                        }
                    }
                }
                updatedMessage = HistoryMessage(
                    id: message.id,
                    content: message.content,
                    images: validImages.isEmpty ? nil : validImages,
                    audio: message.audio,
                    isUser: message.isUser,
                    createdAt: message.createdAt
                )
            }
            
            return updatedMessage
        }
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
