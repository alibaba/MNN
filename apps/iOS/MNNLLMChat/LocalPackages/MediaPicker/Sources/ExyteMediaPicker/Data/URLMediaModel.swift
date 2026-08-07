//
//  SwiftUIView.swift
//  
//
//  Created by Alisa Mylnikova on 12.07.2022.
//

import SwiftUI
import UniformTypeIdentifiers
import AVFoundation

struct URLMediaModel {
    let url: URL
}

extension URLMediaModel: MediaModelProtocol {

    var mediaType: MediaType? {
        if url.isImageFile {
            return .image
        }
        if url.isVideoFile {
            return .video
        }
        return nil
    }

    var duration: CGFloat? {
        CMTimeGetSeconds(AVURLAsset(url: url).duration)
    }

    func getURL() async -> URL? {
        url
    }

    func getThumbnailURL() async -> URL? {
        switch mediaType {
        case .image:
            return url
        case .video:
            return await url.getThumbnailURL()
        case .none:
            return nil
        }
    }

    func getData() async throws -> Data? {
        try? Data(contentsOf: url)
    }

    func getThumbnailData() async -> Data? {
        switch mediaType {
        case .image:
            return try? Data(contentsOf: url)
        case .video:
            return await url.getThumbnailData()
        case .none:
            return nil
        }
    }
}

extension URLMediaModel: Identifiable {
    var id: String {
        url.absoluteString
    }
}

extension URLMediaModel: Equatable {
    static func ==(lhs: URLMediaModel, rhs: URLMediaModel) -> Bool {
        lhs.id == rhs.id
    }
}
