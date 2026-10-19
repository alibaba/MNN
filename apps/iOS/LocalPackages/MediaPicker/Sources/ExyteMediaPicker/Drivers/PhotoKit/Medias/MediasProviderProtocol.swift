//
//  Created by Alex.M on 09.06.2022.
//

import Foundation
import Combine
import Photos
import SwiftUI

protocol MediasProviderProtocol {
    func reload()
    func cancel()

    var assetMediaModelsPublisher: PassthroughSubject<[AssetMediaModel], Never> { get }
}

class BaseMediasProvider: MediasProviderProtocol {
    var selectionParamsHolder: SelectionParamsHolder
    var filterClosure: MediaPicker.FilterClosure?
    var massFilterClosure: MediaPicker.MassFilterClosure?

    @Binding var showingLoadingCell: Bool

    var assetMediaModelsPublisher = PassthroughSubject<[AssetMediaModel], Never>()

    @Published var cancellableTask: Task<Void, Never>?

    private var cancellable: AnyCancellable?
    
    init(selectionParamsHolder: SelectionParamsHolder, filterClosure: MediaPicker.FilterClosure?, massFilterClosure: MediaPicker.MassFilterClosure?, showingLoadingCell: Binding<Bool>) {
        self.selectionParamsHolder = selectionParamsHolder
        self.filterClosure = filterClosure
        self.massFilterClosure = massFilterClosure
        self._showingLoadingCell = showingLoadingCell

        cancellable = photoLibraryChangeLimitedPhotosPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] in
                self?.reload()
            }
    }

    func filterAndPublish(assets: [AssetMediaModel]) {
        if let filterClosure = filterClosure {
            showLoading(true)
            cancellableTask = Task {
                let serialQueue = DispatchQueue(label: "filterSerialQueue")
                var result = [AssetMediaModel]()
                await assets.asyncForEach {
                    if cancellableTask?.isCancelled ?? false {
                        return
                    }
                    if let media = await filterClosure(Media(source: $0)), let model = media.source as? AssetMediaModel {
                        serialQueue.sync {
                            result.append(model)
                            assetMediaModelsPublisher.send(result)
                        }
                    }
                }
                showLoading(false)
            }
        } else if let massFilterClosure = massFilterClosure {
            showLoading(true)
            cancellableTask = Task {
                let result = await massFilterClosure(assets.map { Media(source: $0) })
                assetMediaModelsPublisher.send(result.compactMap { $0.source as? AssetMediaModel })
                showLoading(false)
            }
        }
        else {
            DispatchQueue.main.async { [weak self] in
                self?.assetMediaModelsPublisher.send(assets)
            }
        }
    }

    func showLoading(_ show: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.$showingLoadingCell.wrappedValue = show
        }
    }

    func reload() { }

    func cancel() {
        cancellableTask?.cancel()
    }
}

class MediasProvider {

    static func map(fetchResult: PHFetchResult<PHAsset>, mediaSelectionType: MediaSelectionType) -> [AssetMediaModel] {
        var assetMediaModels: [AssetMediaModel] = []

        if fetchResult.count == 0 {
            return assetMediaModels
        }

        for index in 0...(fetchResult.count - 1) {
            let asset = fetchResult[index]
            if (asset.mediaType == .image && mediaSelectionType.allowsPhoto) || (asset.mediaType == .video && mediaSelectionType.allowsVideo) {
                assetMediaModels.append(AssetMediaModel(asset: asset))
            }
        }
        return assetMediaModels
    }
}
