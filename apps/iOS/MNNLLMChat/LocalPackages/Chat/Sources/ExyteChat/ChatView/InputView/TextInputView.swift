//
//  Created by Alex.M on 14.06.2022.
//

import SwiftUI

struct TextInputView: View {

    @Environment(\.chatTheme) private var theme

    @EnvironmentObject private var globalFocusState: GlobalFocusState

    @Binding var text: String
    var inputFieldId: UUID
    var style: InputViewStyle
    var availableInput: AvailableInputType
    var localization: ChatLocalization

    var body: some View {
        TextField("", text: $text, axis: .vertical)
            .customFocus($globalFocusState.focus, equals: .uuid(inputFieldId))
            .placeholder(when: text.isEmpty) {
                Text(localization.inputPlaceholder)
                    .foregroundColor(theme.colors.inputPlaceholderText)
            }
            .foregroundColor(theme.colors.inputText)
            .padding(.vertical, 10)
            .padding(.leading, !availableInput.isMediaAvailable ? 12 : 0)
            .onTapGesture {
                globalFocusState.focus = .uuid(inputFieldId)
            }
            .tint(theme.colors.sendButtonBackground)
    }
}
