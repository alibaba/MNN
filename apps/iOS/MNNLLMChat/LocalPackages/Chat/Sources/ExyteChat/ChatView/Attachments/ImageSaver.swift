//
//  ImageSaver.swift
//  ExyteChat
//

import SwiftUI
import Photos

/// Shared helper for saving images to the photo library.
final class ImageSaver: ObservableObject {
    @Published var showAlert = false
    @Published var alertMessage = ""

    func saveToPhotoLibrary(url: URL) {
        guard let imageData = try? Data(contentsOf: url),
              let image = UIImage(data: imageData) else {
            alertMessage = NSLocalizedString("Failed to load image.", comment: "Save image error")
            showAlert = true
            return
        }

        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            DispatchQueue.main.async {
                guard let self = self else { return }
                if status == .authorized || status == .limited {
                    UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
                    self.alertMessage = NSLocalizedString("Image saved to Photos.", comment: "Save image success")
                    self.showAlert = true
                } else {
                    self.alertMessage = NSLocalizedString("Photo library access denied. Please enable it in Settings.", comment: "Save image permission denied")
                    self.showAlert = true
                }
            }
        }
    }
}
