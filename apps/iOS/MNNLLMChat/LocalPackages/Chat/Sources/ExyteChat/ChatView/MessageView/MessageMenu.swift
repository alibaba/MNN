//
//  MessageMenu.swift
//
//
//  Created by Alisa Mylnikova on 20.03.2023.
//

import FloatingButton
import enum FloatingButton.Alignment
import SwiftUI

public protocol MessageMenuAction: Equatable, CaseIterable {
    func title() -> String
    func icon() -> Image
}

public enum DefaultMessageMenuAction: MessageMenuAction {
    case copy
    case reply
    case edit(saveClosure: (String) -> Void)

    public func title() -> String {
        switch self {
        case .copy:
            "Copy"
        case .reply:
            "Reply"
        case .edit:
            "Edit"
        }
    }

    public func icon() -> Image {
        switch self {
        case .copy:
            Image(systemName: "doc.on.doc")
        case .reply:
            Image(systemName: "arrowshape.turn.up.left")
        case .edit:
            Image(systemName: "bubble.and.pencil")
        }
    }

    public static func == (lhs: DefaultMessageMenuAction, rhs: DefaultMessageMenuAction) -> Bool {
        switch (lhs, rhs) {
        case (.copy, .copy),
             (.reply, .reply),
             (.edit(_), .edit(_)):
            return true
        default:
            return false
        }
    }

    public static var allCases: [DefaultMessageMenuAction] = [
        .copy, .reply, .edit(saveClosure: { _ in }),
    ]
}

struct MessageMenu<MainButton: View, ActionEnum: MessageMenuAction>: View {
    @Environment(\.chatTheme) private var theme

    @Binding var isShowingMenu: Bool
    @Binding var menuButtonsSize: CGSize
    var alignment: Alignment
    var leadingPadding: CGFloat
    var trailingPadding: CGFloat
    var font: UIFont? = nil
    var onAction: (ActionEnum) -> Void
    var mainButton: () -> MainButton

    var getFont: Font? {
        if let font {
            return Font(font)
        } else {
            return nil
        }
    }

    var body: some View {
        FloatingButton(
            mainButtonView: mainButton().allowsHitTesting(false),
            buttons: ActionEnum.allCases.map {
                menuButton(title: $0.title(), icon: $0.icon(), action: $0)
            },
            isOpen: $isShowingMenu
        )
        .straight()
        // .mainZStackAlignment(.top)
        .initialOpacity(0)
        .direction(.bottom)
        .alignment(alignment)
        .spacing(2)
        .animation(.linear(duration: 0.2))
        .menuButtonsSize($menuButtonsSize)
    }

    func menuButton(title: String, icon: Image, action: ActionEnum) -> some View {
        HStack(spacing: 0) {
            if alignment == .left {
                Color.clear.viewSize(leadingPadding)
            }

            ZStack {
                theme.colors.messageFriendBG
                    .cornerRadius(12)
                HStack {
                    Text(title)
                        .foregroundColor(theme.colors.menuText)
                    Spacer()
                    icon
                        .renderingMode(.template)
                        .foregroundStyle(theme.colors.menuText)
                }
                .font(getFont)
                .padding(.vertical, 11)
                .padding(.horizontal, 12)
            }
            .frame(width: 208)
            .fixedSize()
            .onTapGesture {
                onAction(action)
            }

            if alignment == .right {
                Color.clear.viewSize(trailingPadding)
            }
        }
    }
}
