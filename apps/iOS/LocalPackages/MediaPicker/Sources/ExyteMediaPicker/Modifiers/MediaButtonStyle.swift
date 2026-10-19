//
//  Created by Alex.M on 06.07.2022.
//

import Foundation
import SwiftUI

struct MediaButtonStyle: ButtonStyle {
    func makeBody(configuration: Self.Configuration) -> some View {
        configuration
            .label
            .opacity(configuration.isPressed ? 0.7 : 1.0)
    }
}
