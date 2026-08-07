//
//  Created by Alex.M on 07.06.2022.
//

import Foundation

public enum MediaPickerMode: Equatable {

    case photos
    case albums
    case album(Album)
    case camera
    case cameraSelection

    public static func == (lhs: MediaPickerMode, rhs: MediaPickerMode) -> Bool {
        switch (lhs, rhs) {
        case (.photos, .photos):
            return true
        case (.albums, .albums):
            return true
        case (.album(let a1), .album(let a2)):
            return a1.id == a2.id
        case (.camera, .camera):
            return true
        case (.cameraSelection, .cameraSelection):
            return true
        default:
            return false
        }
    }
}
