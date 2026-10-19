//
//  MessageTextView.swift
//
//
//  Created by Alex.M on 07.07.2022.
//

import MarkdownView
import SwiftUI

public struct MessageTextView: View {
    let text: String?
    let messageUseMarkdown: Bool

    public var body: some View {
        if let text = text, !text.isEmpty {
            textView(text)
        }
    }

    @ViewBuilder
    public func textView(_ text: String) -> some View {
        MarkdownTextViewWrapper(text: text)
    }
}

struct MessageTextView_Previews: PreviewProvider {
    static var previews: some View {
        MessageTextView(text: "Hello world!", messageUseMarkdown: false)
    }
}
