//
//  Created by Alex.M on 08.06.2022.
//

import Foundation
import SwiftUI
import Combine
import Photos

final class SelectionService: ObservableObject {

    var mediaSelectionLimit: Int?
    var onChange: MediaPickerCompletionClosure? = nil

    @Published private(set) var selected: [AssetMediaModel] = []

    var canSendSelected: Bool {
        !selected.isEmpty
    }

    var fitsSelectionLimit: Bool {
        if let selectionLimit = mediaSelectionLimit {
            return selected.count < selectionLimit
        }
        return true
    }

    func canSelect(assetMediaModel: AssetMediaModel) -> Bool {
        fitsSelectionLimit || selected.contains(assetMediaModel)
    }

    func onSelect(assetMediaModel: AssetMediaModel) {
        if let index = selected.firstIndex(of: assetMediaModel) {
            selected.remove(at: index)
        } else {
            if fitsSelectionLimit {
                selected.append(assetMediaModel)
            }
        }
        onChange?(mapToMedia())
    }

    func index(of assetMediaModel: AssetMediaModel) -> Int? {
        selected.firstIndex(of: assetMediaModel)
    }

    func mapToMedia() -> [Media] {
        selected
            .compactMap {
                guard $0.mediaType != nil else {
                    return nil
                }
                return Media(source: $0)
            }
    }

    func removeAll() {
        selected.removeAll()
        onChange?([])
    }

    func updateSelection(with models: [AssetMediaModel]) {
        selected = selected.filter {
            models.contains($0)
        }
    }
}

private extension SelectionService {

    func findAsset(identifier: String) -> Future<PHAsset?, Never> {
        Future { promise in
            let options = PHFetchOptions()
            let photos = PHAsset.fetchAssets(with: options)

            var result: PHAsset?

            photos.enumerateObjects { (asset, _, stop) in
                if asset.localIdentifier == identifier {
                    result = asset
                    stop.pointee = true
                }
            }
            promise(.success(result))
        }
    }
}
