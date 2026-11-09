//
//  Created by Alex.M on 20.06.2022.
//

import Foundation
import Combine
import UIKit

final class ChatViewModel: ObservableObject {

    @Published private(set) var fullscreenAttachmentItem: Optional<Attachment> = nil
    @Published var fullscreenAttachmentPresented = false

    @Published var messageMenuRow: MessageRow?

    let inputFieldId = UUID()

    var didSendMessage: (DraftMessage) -> Void = {_ in}
    var inputViewModel: InputViewModel?
    var globalFocusState: GlobalFocusState?

    func presentAttachmentFullScreen(_ attachment: Attachment) {
        fullscreenAttachmentItem = attachment
        fullscreenAttachmentPresented = true
    }
    
    func dismissAttachmentFullScreen() {
        fullscreenAttachmentPresented = false
        fullscreenAttachmentItem = nil
    }

    func sendMessage(_ message: DraftMessage) {
        didSendMessage(message)
    }

    func messageMenuAction() -> (Message, DefaultMessageMenuAction) -> Void {
        { [weak self] message, action in
            DispatchQueue.main.async {
                self?.messageMenuActionInternal(message: message, action: action)
            }
        }
    }

    @MainActor
    func messageMenuActionInternal(message: Message, action: DefaultMessageMenuAction) {
        switch action {
        case .copy:
            UIPasteboard.general.string = message.text
        case .reply:
            inputViewModel?.attachments.replyMessage = message.toReplyMessage()
            globalFocusState?.focus = .uuid(inputFieldId)
        case .edit(let saveClosure):
            inputViewModel?.text = message.text
            inputViewModel?.edit(saveClosure)
            globalFocusState?.focus = .uuid(inputFieldId)
        }
    }
}
