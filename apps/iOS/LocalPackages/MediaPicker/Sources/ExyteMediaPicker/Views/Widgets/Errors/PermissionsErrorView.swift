//
//  Created by Alex.M on 06.06.2022.
//

import Foundation
import SwiftUI

struct PermissionsErrorView: View {

    let text: String
    let action: (() -> Void)?

    @Environment(\.mediaPickerTheme) private var theme
    
    var body: some View {
        Group {
            if let action = action {
                Button {
                    action()
                } label: {
                    Text(text)
                }
            } else {
                Text(text)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .foregroundColor(theme.error.tint)
        .background(theme.error.background)
        .cornerRadius(5)
        .padding(.horizontal, 20)
    }
}
