//
//  LLMChatVideo.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import ExyteChat
import SwiftUI

struct LLMChatVideo {
    let id: String
    let thumbnail: URL
    let full: URL

    func toChatAttachment() -> Attachment {
        Attachment(
            id: id,
            thumbnail: thumbnail,
            full: full,
            type: .video
        )
    }
}
