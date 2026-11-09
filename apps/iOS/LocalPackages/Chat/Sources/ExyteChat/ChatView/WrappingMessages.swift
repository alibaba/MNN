//
//  WrappingMessages.swift
//
//
//  Created by Alisa Mylnikova on 06.12.2023.
//

import SwiftUI

extension ChatView {
    static func mapMessages(_ messages: [Message], chatType: ChatType, replyMode: ReplyMode) -> [MessagesSection] {
        guard messages.hasUniqueIDs() else {
            fatalError("Messages can not have duplicate ids, please make sure every message gets a unique id")
        }

        let result: [MessagesSection]
        switch replyMode {
        case .quote:
            result = mapMessagesQuoteModeReplies(messages, chatType: chatType, replyMode: replyMode)
        case .answer:
            result = mapMessagesCommentModeReplies(messages, chatType: chatType, replyMode: replyMode)
        }

        return result
    }

    static func mapMessagesQuoteModeReplies(_ messages: [Message], chatType: ChatType, replyMode: ReplyMode) -> [MessagesSection] {
        let dates = Set(messages.map { $0.createdAt.startOfDay() })
            .sorted()
            .reversed()
        var result: [MessagesSection] = []

        for date in dates {
            let section = MessagesSection(
                date: date,
                // use fake isFirstSection/isLastSection because they are not needed for quote replies
                rows: wrapSectionMessages(messages.filter { $0.createdAt.isSameDay(date) }, chatType: chatType, replyMode: replyMode, isFirstSection: false, isLastSection: false)
            )
            result.append(section)
        }

        return result
    }

    static func mapMessagesCommentModeReplies(_ messages: [Message], chatType: ChatType, replyMode: ReplyMode) -> [MessagesSection] {
        let firstLevelMessages = messages.filter { m in
            m.replyMessage == nil
        }

        let dates = Set(firstLevelMessages.map { $0.createdAt.startOfDay() })
            .sorted()
            .reversed()
        var result: [MessagesSection] = []

        for date in dates {
            let dayFirstLevelMessages = firstLevelMessages.filter { $0.createdAt.isSameDay(date) }
            var dayMessages = [Message]() // insert second level in between first level
            for m in dayFirstLevelMessages {
                var replies = getRepliesFor(id: m.id, messages: messages)
                replies.sort { $0.createdAt < $1.createdAt }
                if chatType == .conversation {
                    dayMessages.append(m)
                }
                dayMessages.append(contentsOf: replies)
                if chatType == .comments {
                    dayMessages.append(m)
                }
            }

            let isFirstSection = dates.first == date
            let isLastSection = dates.last == date
            let sectionRows = wrapSectionMessages(dayMessages, chatType: chatType, replyMode: replyMode, isFirstSection: isFirstSection, isLastSection: isLastSection)
            result.append(MessagesSection(date: date, rows: sectionRows))
        }

        return result
    }

    private static func getRepliesFor(id: String, messages: [Message]) -> [Message] {
        messages.compactMap { m in
            if m.replyMessage?.id == id {
                return m
            }
            return nil
        }
    }

    private static func wrapSectionMessages(_ messages: [Message], chatType: ChatType, replyMode: ReplyMode, isFirstSection: Bool, isLastSection: Bool) -> [MessageRow] {
        messages
            .enumerated()
            .map {
                let index = $0.offset
                let message = $0.element
                let nextMessage = chatType == .conversation ? messages[safe: index + 1] : messages[safe: index - 1]
                let prevMessage = chatType == .conversation ? messages[safe: index - 1] : messages[safe: index + 1]

                let nextMessageExists = nextMessage != nil
                let prevMessageExists = prevMessage != nil
                let nextMessageIsSameUser = nextMessage?.user.id == message.user.id
                let prevMessageIsSameUser = prevMessage?.user.id == message.user.id

                let position: PositionInUserGroup
                if nextMessageExists, nextMessageIsSameUser, prevMessageIsSameUser {
                    position = .middle
                } else if !nextMessageExists || !nextMessageIsSameUser, !prevMessageIsSameUser {
                    position = .single
                } else if nextMessageExists, nextMessageIsSameUser {
                    position = .first
                } else {
                    position = .last
                }

                if replyMode == .quote {
                    return MessageRow(message: $0.element, positionInUserGroup: position, commentsPosition: nil)
                }

                let nextMessageIsAReply = nextMessage?.replyMessage != nil
                let nextMessageIsFirstLevel = nextMessage?.replyMessage == nil
                let prevMessageIsFirstLevel = prevMessage?.replyMessage == nil

                let positionInComments: PositionInCommentsGroup
                if message.replyMessage == nil, !nextMessageIsAReply {
                    positionInComments = .singleFirstLevelPost
                } else if message.replyMessage == nil, nextMessageIsAReply {
                    positionInComments = .firstLevelPostWithComments
                } else if nextMessageIsFirstLevel {
                    positionInComments = .lastComment
                } else if prevMessageIsFirstLevel {
                    positionInComments = .firstComment
                } else {
                    positionInComments = .middleComment
                }

                let positionInSection: PositionInSection
                if !prevMessageExists, !nextMessageExists {
                    positionInSection = .single
                } else if !prevMessageExists {
                    positionInSection = .first
                } else if !nextMessageExists {
                    positionInSection = .last
                } else {
                    positionInSection = .middle
                }

                let positionInChat: PositionInChat
                if !isFirstSection, !isLastSection {
                    positionInChat = .middle
                } else if !prevMessageExists, !nextMessageExists, isFirstSection, isLastSection {
                    positionInChat = .single
                } else if !prevMessageExists, isFirstSection {
                    positionInChat = .first
                } else if !nextMessageExists, isLastSection {
                    positionInChat = .last
                } else {
                    positionInChat = .middle
                }

                let commentsPosition = CommentsPosition(inCommentsGroup: positionInComments, inSection: positionInSection, inChat: positionInChat)

                return MessageRow(message: $0.element, positionInUserGroup: position, commentsPosition: commentsPosition)
            }
            .reversed()
    }
}
