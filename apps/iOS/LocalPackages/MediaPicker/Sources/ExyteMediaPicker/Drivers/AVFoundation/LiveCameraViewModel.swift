//
//  Created by Alex.M on 07.06.2022.
//

import Foundation
import AVFoundation
import CoreImage
import UIKit.UIImage

/**
 Object with live camera preview. Use `captureImage: UIImage` publisher.
 
 Used solution (with adapt for new iOs version): https://medium.com/ios-os-x-development/ios-camera-frames-extraction-d2c0f80ed05a
 */
class LiveCameraViewModel: NSObject, ObservableObject {
    
    private let quality = AVCaptureSession.Preset.medium

    private let sessionQueue = DispatchQueue(label: "LiveCameraPreviewQueue")
    let captureSession = AVCaptureSession()

    @Published public var capturedImage: UIImage = UIImage()

    override init() {
        super.init()
        sessionQueue.async { [weak self] in
            self?.configureSession()
            self?.captureSession.startRunning()
        }
    }

    func startSession() {
        sessionQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
    }

    deinit {
        captureSession.stopRunning()
    }

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = quality
        guard let captureDevice = captureDevice else { return }
        guard let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        guard captureSession.canAddInput(captureDeviceInput) else { return }
        captureSession.addInput(captureDeviceInput)
        captureSession.commitConfiguration()
    }

    private var captureDevice: AVCaptureDevice? {
        AVCaptureDevice.default(for: .video)
    }
}
