//
//  LLMChatUser.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import SwiftUI
import ExyteChat

struct LLMChatUser: Equatable {
    let uid: String
    let name: String
    let avatar: URL?

    init(uid: String, name: String, avatar: URL? = nil) {
        self.uid = uid
        self.name = name
        self.avatar = avatar
    }
}

extension LLMChatUser {
    var isCurrentUser: Bool {
        uid == "1"
    }
}

extension LLMChatUser {
    func toChatUser() -> ExyteChat.User {
        ExyteChat.User(id: uid, name: name, avatarURL: avatar, isCurrentUser: isCurrentUser)
    }
}
