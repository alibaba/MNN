//
//  LiveCameraView.swift
//  
//
//  Created by Alexandra Afonasova on 18.10.2022.
//

import SwiftUI
import AVFoundation

public struct LiveCameraView: UIViewRepresentable {

    let session: AVCaptureSession
    var videoGravity: AVLayerVideoGravity = .resizeAspect
    var orientation: UIDeviceOrientation = UIDevice.current.orientation

    public func makeUIView(context: Context) -> LiveVideoCaptureView {
        NotificationCenter.default.post(
            name: cameraChangePermissionNotification,
            object: nil)

        return LiveVideoCaptureView(
            session: session,
            videoGravity: videoGravity,
            orientation: orientation
        )
    }

    public func updateUIView(_ uiView: LiveVideoCaptureView, context: Context) {
        uiView.updateOrientation(orientation)
    }

}

public final class LiveVideoCaptureView: UIView {

    var session: AVCaptureSession? {
        get {
            return videoLayer.session
        }
        set (session) {
            videoLayer.session = session
        }
    }

    public override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    private var videoLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    init(
        frame: CGRect = .zero,
        session: AVCaptureSession? = nil,
        videoGravity: AVLayerVideoGravity = .resizeAspect,
        orientation: UIDeviceOrientation
    ) {
        super.init(frame: frame)
        self.session = session
        videoLayer.videoGravity = videoGravity
        updateOrientation(orientation)
    }

    func updateOrientation(_ orientation: UIDeviceOrientation) {
        guard let connection = videoLayer.connection, connection.isVideoOrientationSupported else { return }
        connection.videoOrientation = AVCaptureVideoOrientation(orientation)
    }

}
