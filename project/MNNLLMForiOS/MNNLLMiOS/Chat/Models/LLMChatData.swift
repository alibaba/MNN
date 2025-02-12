//
//  LLMChatData.swift
//  MNNLLMiOS
//  Modified from ExyteChat's Chat Example
// 
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import UIKit
import ExyteChat
import ExyteMediaPicker

final class LLMChatData {
    var assistant: LLMChatUser
    var system: LLMChatUser
    
    let user = LLMChatUser(
        uid: "1",
        name: "user",
        avatar: AssetExtractor.createLocalUrl(forImageNamed: "mnn_icon", withExtension: "png")
    )
    
    init(modelInfo: ModelInfo) {
        let icon = ModelIconManager.shared.getModelImage(with: modelInfo.localPath) ?? "mnn_icon"
        
        self.assistant = LLMChatUser(
            uid: "2",
            name: modelInfo.name,
            avatar: AssetExtractor.createLocalUrl(forImageNamed: icon, withExtension: "png")
        )
        
        self.system = LLMChatUser(
            uid: "0",
            name: modelInfo.name,
            avatar: AssetExtractor.createLocalUrl(forImageNamed: icon, withExtension: "png")
        )
    }
    
    func greatingMessage(historyMessages: [HistoryMessage]?) -> [LLMChatMessage] {
        let sender = system
        let date = Date()
        
        var allHistoryMessages: [LLMChatMessage] = []
        if let msgs = historyMessages {
            for msg in msgs {
                var sender = assistant
                if msg.isUser == true {
                    sender = user
                }
                
                allHistoryMessages.append(
                    LLMChatMessage(uid: UUID().uuidString,
                                   sender: sender,
                                   createdAt: msg.createdAt,
                                   text: msg.content,
                                   images: msg.images ?? [],
                                   videos: [],
                                   recording: msg.audio,
                                   replyMessage: nil))
            }
            return allHistoryMessages
        }
        
        
        return [LLMChatMessage(
            uid: UUID().uuidString,
            sender: sender,
            createdAt: date,
            status: sender.isCurrentUser ? .read : nil,
            text: NSLocalizedString("WelcomeSceneText", comment: "") + sender.name + "!",
            images: [],
            videos: [],
            recording: nil,
            replyMessage: nil
        )]
    }
}
