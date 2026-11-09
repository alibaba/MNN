//
//  Created by Alex.M on 08.06.2022.
//

import Foundation
import Combine
import AVFoundation
import Photos

final class PermissionsService: ObservableObject {
    @Published var cameraAction: CameraAction? = .authorize
    @Published var photoLibraryAction: PhotoLibraryAction? = .authorize

    private var subscriptions = Set<AnyCancellable>()

    init() {
        photoLibraryChangePermissionPublisher
            .sink { [weak self] in
                self?.checkPhotoLibraryAuthorizationStatus()
            }
            .store(in: &subscriptions)

        cameraChangePermissionPublisher
            .sink { [weak self] in
                self?.checkCameraAuthorizationStatus()
            }
            .store(in: &subscriptions)
    }

    /// photoLibraryChangePermissionPublisher gets called multiple times even when nothing changed in photo library, so just use this one to make sure the closure runs exactly once
    static func requestPhotoLibraryPermission(_ permissionGrantedClosure: @escaping ()->()) {
        PHPhotoLibrary.requestAuthorization { status in
            if status == .authorized || status == .limited {
                permissionGrantedClosure()
            }
        }
    }

    func requestCameraPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] _ in
            self?.checkCameraAuthorizationStatus()
        }
    }

    func checkPhotoLibraryAuthorizationStatus() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        
        var result: PhotoLibraryAction?
        switch status {
        case .restricted:
            // TODO: Make sure that access can't change when status == .restricted
            result = .unavailable
        case .notDetermined, .denied:
            result = .authorize
        case .authorized:
            // Do nothing
            break
        case .limited:
            result = .selectMore
        @unknown default:
            result = .unknown
        }

        DispatchQueue.main.async { [weak self] in
            self?.photoLibraryAction = result
        }
    }

    func checkCameraAuthorizationStatus() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        var result: CameraAction?
#if targetEnvironment(simulator)
        result = .unavailable
#else
        switch status {
        case .restricted:
            result = .unavailable
        case .notDetermined, .denied:
            result = .authorize
        case .authorized:
            // Do nothing
            break
        @unknown default:
            result = .unknown
        }
#endif
        DispatchQueue.main.async { [weak self] in
            self?.cameraAction = result
        }
    }
}

extension PermissionsService {
    enum CameraAction {
        case authorize
        case unavailable
        case unknown
    }

    enum PhotoLibraryAction {
        case selectMore
        case authorize
        case unavailable
        case unknown
    }
}
