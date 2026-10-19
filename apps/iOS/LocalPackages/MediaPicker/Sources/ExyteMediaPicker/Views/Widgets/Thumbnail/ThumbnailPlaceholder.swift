//
//  Created by Alex.M on 30.05.2022.
//

import SwiftUI

struct ThumbnailPlaceholder: View {

    var body: some View {
        Rectangle()
            .fill(.gray.opacity(0.3))
            .aspectRatio(1, contentMode: .fill)
    }
}
