//
//  Created by Alex.M on 09.06.2022.
//

import Foundation

import Foundation
import Photos
import Combine
import SwiftUI

final class AllPhotosProvider: BaseMediasProvider {

    override func reload() {
        PermissionsService.requestPhotoLibraryPermission { [ weak self] in
            self?.reloadInternal()
        }
    }

    func reloadInternal() {
        let allPhotosOptions = PHFetchOptions()
        allPhotosOptions.sortDescriptors = [
            NSSortDescriptor(key: "creationDate", ascending: false)
        ]
        let allPhotos = PHAsset.fetchAssets(with: allPhotosOptions)
        let assets = MediasProvider.map(fetchResult: allPhotos, mediaSelectionType: selectionParamsHolder.mediaType)
        filterAndPublish(assets: assets)
    }
}
