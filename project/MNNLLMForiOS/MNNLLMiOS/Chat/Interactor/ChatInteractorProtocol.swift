//
//  ChatInteractorProtocol.swift
//  MNNLLMiOS
//  Modified from ExyteChat's Chat Example
//  Created by 游薪渝(揽清) on 2025/1/15.
//

import Combine
import ExyteChat

protocol ChatInteractorProtocol {
    var messages: AnyPublisher<[LLMChatMessage], Never> { get }
    var senders: [LLMChatUser] { get }
    var otherSenders: [LLMChatUser] { get }

    func send(draftMessage: ExyteChat.DraftMessage, userType: UserType)

    func connect()
    func disconnect()

    func loadNextPage() -> Future<Bool, Never>
}
