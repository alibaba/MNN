//
//  Created by Alex.M on 22.06.2022.
//

import Foundation
import Combine

final class FullscreenMediaPagesViewModel: ObservableObject {
    var attachments: [Attachment]
    @Published var index: Int

    @Published var showMinis = true
    @Published var offset: CGSize = .zero

    @Published var videoPlaying = false
    @Published var videoMuted = false

    @Published var toggleVideoPlaying = {}
    @Published var toggleVideoMuted = {}

    init(attachments: [Attachment], index: Int) {
        self.attachments = attachments
        self.index = index
    }
}
