//
//  Created by Alex.M on 27.05.2022.
//

import Foundation
import Photos

struct AssetMediaModel {
    let asset: PHAsset
}

extension AssetMediaModel: MediaModelProtocol {

    var mediaType: MediaType? {
        switch asset.mediaType {
        case .image:
            return .image
        case .video:
            return .video
        default:
            return nil
        }
    }

    var duration: CGFloat? {
        CGFloat(asset.duration)
    }

    func getURL() async -> URL? {
        await asset.getURL()
    }

    func getThumbnailURL() async -> URL? {
        await asset.getThumbnailURL()
    }

    func getData() async throws -> Data? {
        try await asset.getData()
    }

    func getThumbnailData() async -> Data? {
        await asset.getThumbnailData()
    }
}

extension AssetMediaModel: Identifiable {
    var id: String {
        asset.localIdentifier
    }
}

extension AssetMediaModel: Equatable {
    static func ==(lhs: AssetMediaModel, rhs: AssetMediaModel) -> Bool {
        lhs.id == rhs.id
    }
}
