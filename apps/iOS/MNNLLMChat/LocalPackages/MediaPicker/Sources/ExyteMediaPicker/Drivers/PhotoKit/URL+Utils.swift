//
//  SwiftUIView.swift
//  
//
//  Created by Alisa Mylnikova on 21.04.2023.
//

import SwiftUI
import Photos

extension URL {

    func getThumbnailURL() async -> URL? {
        let asset: AVAsset = AVAsset(url: self)
        if let thumbnailData = asset.generateThumbnail() {
            return FileManager.storeToTempDir(data: thumbnailData)
        }
        return nil
    }

    func getThumbnailData() async -> Data? {
        let asset: AVAsset = AVAsset(url: self)
        return asset.generateThumbnail()
    }
    
    var isImageFile: Bool {
        UTType(filenameExtension: pathExtension)?.conforms(to: .image) ?? false
    }

    var isVideoFile: Bool {
        UTType(filenameExtension: pathExtension)?.conforms(to: .audiovisualContent) ?? false
    }
}
