//
//  FilterMediaPicker.swift
//  Example-iOS
//
//  Created by Alisa Mylnikova on 18.05.2023.
//

import Foundation
import SwiftUI
import ExyteMediaPicker

struct FilterMediaPicker: View {

    @Binding var isPresented: Bool
    @Binding var medias: [Media]

    var body: some View {
        MediaPicker(
            isPresented: $isPresented,
            onChange: { medias = $0 }
        )
        .applyFilter { await isMostlyBlueAndGreen($0) }
    }

    private func isMostlyBlueAndGreen(_ media: Media) async -> Media? {
        guard let data = await media.getThumbnailData() else { return nil }
        guard let uiImage = UIImage(data: data) else { return nil }

        let color = uiImage.averageColor
        var red: CGFloat = 0.0, green: CGFloat = 0.0, blue: CGFloat = 0.0, alpha: CGFloat = 0.0
        color?.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        if blue > red, green > red {
           return media
        } else {
            return nil
        }
    }
}

extension UIImage {
    var averageColor: UIColor? {
        guard let inputImage = CIImage(image: self) else { return nil }
        let extentVector = CIVector(x: inputImage.extent.origin.x, y: inputImage.extent.origin.y, z: inputImage.extent.size.width, w: inputImage.extent.size.height)

        guard let filter = CIFilter(name: "CIAreaAverage", parameters: [kCIInputImageKey: inputImage, kCIInputExtentKey: extentVector]) else { return nil }
        guard let outputImage = filter.outputImage else { return nil }

        var bitmap = [UInt8](repeating: 0, count: 4)
        let context = CIContext(options: [.workingColorSpace: kCFNull as Any])
        context.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)

        return UIColor(red: CGFloat(bitmap[0]) / 255, green: CGFloat(bitmap[1]) / 255, blue: CGFloat(bitmap[2]) / 255, alpha: CGFloat(bitmap[3]) / 255)
    }
}
