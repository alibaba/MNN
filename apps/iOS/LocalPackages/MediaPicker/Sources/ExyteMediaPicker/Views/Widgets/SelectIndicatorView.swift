//
//  Created by Alex.M on 27.05.2022.
//

import SwiftUI

struct SelectIndicatorView: View {

    @EnvironmentObject private var selectionService: SelectionService

    @Environment(\.mediaPickerTheme) var theme

    var index: Int?
    var isFullscreen: Bool
    var canSelect: Bool
    var selectionParamsHolder: SelectionParamsHolder

    var body: some View {
        Group {
            switch selectionParamsHolder.selectionStyle {
            case .checkmark:
                checkView
            case .count:
                countView
            }
        }
        .frame(width: 24, height: 24)
        .padding(.top, isFullscreen ? 11 : -2)
    }

    var checkView: some View {
        Group {
            if index != nil {
                Image(systemName: "checkmark.circle.fill")
                    .resizable()
                    .foregroundColor(theme.selection.selectedTint)
                    .padding(2)
                    .background {
                        Circle()
                            .fill(theme.selection.selectedBackground)
                    }
            } else if canSelect {
                Image(systemName: "circle")
                    .resizable()
                    .foregroundColor(isFullscreen ? theme.selection.fullscreenTint : theme.selection.emptyTint)
                    .background {
                        Circle()
                            .fill(theme.selection.emptyBackground)
                    }
            }
        }
    }
    
    var countView: some View {
        Group {
            if let index = index {
                Image(systemName: "\(index + 1).circle.fill")
                    .resizable()
                    .foregroundColor(theme.selection.selectedTint)
                    .background {
                        Circle()
                            .fill(theme.selection.selectedBackground)
                    }
            } else if canSelect {
                Image(systemName: "circle")
                    .resizable()
                    .foregroundColor(isFullscreen ? theme.selection.fullscreenTint : theme.selection.emptyTint)
                    .background {
                        Circle()
                            .fill(theme.selection.emptyBackground)
                    }
            }
        }
    }
}
