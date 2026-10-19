//
//  LLMChatData.swift
//  MNNLLMiOS
//  Modified from ExyteChat's Chat Example
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import Foundation
import ExyteChat
import UIKit

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

        assistant = LLMChatUser(
            uid: "2",
            name: modelInfo.name,
            avatar: AssetExtractor.createLocalUrl(forImageNamed: icon, withExtension: "png")
        )

        system = LLMChatUser(
            uid: "0",
            name: modelInfo.name,
            avatar: AssetExtractor.createLocalUrl(forImageNamed: icon, withExtension: "png")
        )
    }

    func greatingMessage(historyMessages: [HistoryMessage]?) -> [LLMChatMessage] {
        var allHistoryMessages: [LLMChatMessage] = []
        if let msgs = historyMessages {
            for msg in msgs {
                var sender = assistant
                if msg.isUser == true {
                    sender = user
                } else if msg.role == "0" {
                    sender = system
                }

                allHistoryMessages.append(
                    LLMChatMessage(uid: msg.id,
                                   sender: sender,
                                   createdAt: msg.createdAt,
                                   useMarkdown: msg.useMarkdown ?? true,
                                   text: msg.content,
                                   thinkText: msg.thinkText ?? "",
                                   performanceData: msg.performanceData,
                                   toolSteps: msg.toolSteps ?? [],
                                   images: msg.images ?? [],
                                   videos: [],
                                   recording: msg.audio,
                                   replyMessage: nil))
            }
            return allHistoryMessages
        }

        // Pocket shows its welcome screen (branding + candidate prompts) on
        // fresh conversations instead of a greeting bubble.
        guard MNNChatConfiguration.current.showsGreetingBubble else {
            return []
        }

        return [LLMChatMessage(
            uid: UUID().uuidString,
            sender: system,
            createdAt: Date(),
            status: system.isCurrentUser ? .read : nil,
            useMarkdown: false,
            text: AppText.text("WelcomeSceneText", comment: "") + system.name + "!",
            images: [],
            videos: [],
            recording: nil,
            replyMessage: nil
        )]
    }
}
