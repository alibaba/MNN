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
                        
                        // 验证最终文件是否存在
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
                var historyMessages = try JSONDecoder().decode([HistoryMessage].self, from: messagesData)
                
                // 验证并修复图片路径
                historyMessages = validateAndFixImagePaths(historyMessages, historyId: history[id])
                
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
    
    // 验证并修复历史消息中的图片路径
    private func validateAndFixImagePaths(_ messages: [HistoryMessage], historyId: String) -> [HistoryMessage] {
        return messages.map { message in
            var updatedMessage = message
            
            if let images = message.images {
                let validImages = images.compactMap { image -> LLMChatImage? in
                    // 检查图片文件是否存在
                    if ChatHistoryFileManager.shared.validateFileExists(at: image.full) {
                        return image
                    } else {
                        // 尝试通过文件名重新构建路径
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
