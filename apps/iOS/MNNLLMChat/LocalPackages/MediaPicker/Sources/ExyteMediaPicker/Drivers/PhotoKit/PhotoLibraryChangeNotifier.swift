//
//  Created by Alex.M on 08.06.2022.
//

import Foundation
import Combine
import Photos

let photoLibraryChangePermissionNotification = Notification.Name(rawValue: "PhotoLibraryChangePermissionNotification")

let photoLibraryChangeLimitedPhotosNotification = Notification.Name(rawValue: "PhotoLibraryChangeLimitedPhotosNotification")

let cameraChangePermissionNotification = Notification.Name(rawValue: "cameraChangePermissionNotification")

let photoLibraryChangePermissionPublisher = NotificationCenter.default
    .publisher(for: photoLibraryChangePermissionNotification)
    .map { _ in }
    .share()

let photoLibraryChangeLimitedPhotosPublisher = NotificationCenter.default
    .publisher(for: photoLibraryChangeLimitedPhotosNotification)
    .map { _ in }
    .share()

let cameraChangePermissionPublisher = NotificationCenter.default
    .publisher(for: cameraChangePermissionNotification)
    .map { _ in }
    .share()

final class PhotoLibraryChangePermissionWatcher: NSObject, PHPhotoLibraryChangeObserver {
    override init() {
        super.init()
        PHPhotoLibrary.shared().register(self)
    }

    deinit {
        PHPhotoLibrary.shared().unregisterChangeObserver(self)
    }

    func photoLibraryDidChange(_ changeInstance: PHChange) {
        // gets called too often, even if nothing changed - a bug?
        NotificationCenter.default.post(
            name: photoLibraryChangePermissionNotification,
            object: nil)
    }
}
