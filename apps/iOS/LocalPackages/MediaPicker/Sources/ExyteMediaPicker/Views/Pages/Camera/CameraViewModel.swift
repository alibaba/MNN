//
//  CameraViewModel.swift
//  
//
//  Created by Alexandra Afonasova on 18.10.2022.
//

import Foundation
import AVFoundation
import Combine
import UIKit
import SwiftUI

final class CameraViewModel: NSObject, ObservableObject {

    struct CaptureDevice {
        let device: AVCaptureDevice
        let position: AVCaptureDevice.Position
        let defaultZoom: CGFloat
        let maxZoom: CGFloat
    }

    @Published private(set) var flashEnabled = false
    @Published private(set) var snapOverlay = false

    let captureSession = AVCaptureSession()
    var capturedPhotoPublisher: AnyPublisher<URL, Never> { capturedPhotoSubject.eraseToAnyPublisher() }

    private let photoOutput = AVCapturePhotoOutput()
    private let videoOutput = AVCaptureMovieFileOutput()
    private let motionManager = MotionManager()
    private let sessionQueue = DispatchQueue(label: "LiveCameraQueue")
    private let capturedPhotoSubject = PassthroughSubject<URL, Never>()
    private var captureDevice: CaptureDevice?
    private var lastPhotoActualOrientation: UIDeviceOrientation?

    private let minScale: CGFloat = 1
    private let singleCameraMaxScale: CGFloat = 5
    private let dualCameraMaxScale: CGFloat = 8
    private let tripleCameraMaxScale: CGFloat = 12
    private var lastScale: CGFloat = 1
    private var zoomAllowed: Bool { captureDevice?.position == .back }

    override init() {
        super.init()
        sessionQueue.async { [weak self] in
            self?.configureSession()
            self?.captureSession.startRunning()
        }
    }

    deinit {
        captureSession.stopRunning()
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

    func takePhoto() {
        let settings = AVCapturePhotoSettings()
        settings.flashMode = flashEnabled ? .on : .off
        photoOutput.capturePhoto(with: settings, delegate: self)
        lastPhotoActualOrientation = motionManager.orientation

        withAnimation(.linear(duration: 0.1)) { snapOverlay = true }
        withAnimation(.linear(duration: 0.1).delay(0.1)) { snapOverlay = false }
    }

    func startVideoCapture() {
        setVideoTorchMode(flashEnabled ? .on : .off)

        let videoUrl = FileManager.getTempUrl()
        videoOutput.startRecording(to: videoUrl, recordingDelegate: self)
    }

    func stopVideoCapture() {
        setVideoTorchMode(.off)
        videoOutput.stopRecording()
    }

    func setVideoTorchMode(_ mode: AVCaptureDevice.TorchMode) {
        if captureDevice?.device.torchMode != mode {
            try? captureDevice?.device.lockForConfiguration()
            captureDevice?.device.torchMode = mode
            captureDevice?.device.unlockForConfiguration()
        }
    }

    func flipCamera() {
        sessionQueue.async { [weak self] in
            guard let session = self?.captureSession, let input = session.inputs.first else {
                return
            }

            let newPosition: AVCaptureDevice.Position = self?.captureDevice?.position == .back ? .front : .back

            session.beginConfiguration()
            session.removeInput(input)
            self?.addInput(to: session, for: newPosition)
            session.commitConfiguration()
        }
    }

    func toggleFlash() {
        flashEnabled.toggle()
    }

    func zoomChanged(_ scale: CGFloat) {
        if !zoomAllowed { return }
        zoomCamera(resolveScale(scale))
    }

    func zoomEnded(_ scale: CGFloat) {
        if !zoomAllowed { return }

        lastScale = resolveScale(scale)
        zoomCamera(lastScale)
    }

    private func resolveScale(_ gestureScale: CGFloat) -> CGFloat {
        let newScale = lastScale * gestureScale
        let maxScale = captureDevice?.maxZoom ?? singleCameraMaxScale
        return max(min(maxScale, newScale), minScale)
    }

    private func zoomCamera(_ scale: CGFloat) {
        do {
            try captureDevice?.device.lockForConfiguration()
            captureDevice?.device.videoZoomFactor = scale
            captureDevice?.device.unlockForConfiguration()
        } catch {}
    }

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo
        addInput(to: captureSession)
        addOutput(to: captureSession)
        captureSession.commitConfiguration()
    }

    private func addInput(to session: AVCaptureSession, for position: AVCaptureDevice.Position = .back) {
        guard let captureDevice = selectCaptureDevice(for: position) else { return }
        guard let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
        guard session.canAddInput(captureDeviceInput) else { return }
        session.addInput(captureDeviceInput)

        guard let captureAudioDevice = selectAudioCaptureDevice() else { return }
        guard let captureAudioDeviceInput = try? AVCaptureDeviceInput(device: captureAudioDevice) else { return }
        guard session.canAddInput(captureAudioDeviceInput) else { return }
        session.addInput(captureAudioDeviceInput)

        let defaultZoom = CGFloat(truncating: captureDevice.virtualDeviceSwitchOverVideoZoomFactors.first ?? minScale as NSNumber)

        let maxZoom: CGFloat
        let cameraCount = captureDevice.virtualDeviceSwitchOverVideoZoomFactors.count + 1
        switch cameraCount {
        case 1: maxZoom = singleCameraMaxScale
        case 2: maxZoom = dualCameraMaxScale
        default: maxZoom = tripleCameraMaxScale
        }

        let device = CaptureDevice(
            device: captureDevice,
            position: position,
            defaultZoom: defaultZoom,
            maxZoom: maxZoom
        )
        self.captureDevice = device

        if position == .back {
            captureDeviceInput.device.videoZoomFactor = device.defaultZoom
            lastScale = device.defaultZoom
        }
    }

    private func addOutput(to session: AVCaptureSession) {
        photoOutput.isLivePhotoCaptureEnabled = false
        guard session.canAddOutput(photoOutput) else { return }
        session.addOutput(photoOutput)

        guard session.canAddOutput(videoOutput) else { return }
        session.addOutput(videoOutput)

        updateOutputOrientation()
    }

    private func updateOutputOrientation() {
        guard let connection = photoOutput.connection(with: .video), connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = .portrait
    }

    private func selectCaptureDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: [
                .builtInDualCamera,
                .builtInDualWideCamera,
                .builtInTripleCamera,
                .builtInTelephotoCamera,
                .builtInTrueDepthCamera,
                .builtInUltraWideCamera,
                .builtInWideAngleCamera
            ],
            mediaType: .video,
            position: position)

        if let camera = session.devices.first(where: { $0.deviceType == .builtInTripleCamera }) {
            return camera
        } else if let camera = session.devices.first(where: { $0.deviceType == .builtInDualCamera }) {
            return camera
        } else {
            return session.devices.first
        }
    }

    private func selectAudioCaptureDevice() -> AVCaptureDevice? {
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInMicrophone],
            mediaType: .audio,
            position: .unspecified)

        return session.devices.first
    }

}

extension CameraViewModel: AVCapturePhotoCaptureDelegate {
    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        guard let cgImage = photo.cgImageRepresentation() else { return }

        let photoOrientation: UIImage.Orientation
        if let orientation = lastPhotoActualOrientation {
            photoOrientation = UIImage.Orientation(orientation)
        } else {
            photoOrientation = UIImage.Orientation.default
        }

        guard let data = UIImage(
            cgImage: cgImage,
            scale: 1,
            orientation: photoOrientation
        ).jpegData(compressionQuality: 0.8) else { return }

        capturedPhotoSubject.send(FileManager.storeToTempDir(data: data))
    }
}

extension CameraViewModel: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        capturedPhotoSubject.send(outputFileURL)
    }
}
