//
//  FirestoreMessage.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 12.07.2023.
//

import ExyteChat
import FirebaseFirestore
import FirebaseFirestoreSwift
import Foundation

public struct FirestoreMessage: Codable, Hashable {
    @DocumentID public var id: String?
    public var userId: String
    @ServerTimestamp public var createdAt: Date?

    public var text: String
    public var attachments: [FirestoreAttachment]
    public var recording: FirestoreRecording?
    public var replyMessage: FirestoreReply?
}

public struct FirestoreAttachment: Codable, Hashable {
    public let thumbURL: String
    public let url: String
    public let type: AttachmentType
}

public struct FirestoreReply: Codable, Hashable {
    @DocumentID public var id: String?
    public var userId: String
    @ServerTimestamp public var createdAt: Date?

    public var text: String
    public var attachments: [FirestoreAttachment]
    public var recording: FirestoreRecording?
}

public struct FirestoreRecording: Codable, Hashable {
    public var duration: CGFloat
    public var waveformSamples: [CGFloat]
    public var url: String
}
