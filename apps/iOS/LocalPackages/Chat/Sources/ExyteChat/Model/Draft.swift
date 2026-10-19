//
//  Created by Alex.M on 17.06.2022.
//

import Foundation
import ExyteMediaPicker

public struct DraftMessage {
    public var id: String?
    public let text: String
    public var thinkText: String?
    public var performanceData: String?
    public var useMarkdown: Bool = true
    public let medias: [Media]
    public let recording: Recording?
    public let replyMessage: ReplyMessage?
    public let createdAt: Date

    public init(id: String? = nil, 
                text: String,
                thinkText: String?,
                performanceData: String? = nil,
                useMarkdown: Bool = true,
                medias: [Media],
                recording: Recording?,
                replyMessage: ReplyMessage?,
                createdAt: Date) {
        self.id = id
        self.text = text
        self.thinkText = thinkText
        self.performanceData = performanceData
        self.useMarkdown = useMarkdown
        self.medias = medias
        self.recording = recording
        self.replyMessage = replyMessage
        self.createdAt = createdAt
    }
}
