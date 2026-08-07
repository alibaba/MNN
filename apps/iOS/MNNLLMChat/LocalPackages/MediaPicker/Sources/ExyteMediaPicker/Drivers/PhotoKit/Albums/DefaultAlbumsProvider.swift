//
//  Created by Alex.M on 10.06.2022.
//

import Foundation
import Combine
import Photos

final class DefaultAlbumsProvider: AlbumsProviderProtocol {

    private var subject = CurrentValueSubject<[AlbumModel], Never>([])
    private var albumsCancellable: AnyCancellable?
    private var permissionCancellable: AnyCancellable?
    
    var albums: AnyPublisher<[AlbumModel], Never> {
        subject.eraseToAnyPublisher()
    }

    var mediaSelectionType: MediaSelectionType = .photoAndVideo

    func reload() {
        PermissionsService.requestPhotoLibraryPermission { [ weak self] in
            self?.reloadInternal()
        }
    }

    func reloadInternal() {
        albumsCancellable = [PHAssetCollectionType.album, .smartAlbum]
            .publisher
            .map { fetchAlbums(type: $0) }
            .scan([], +)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] in
                self?.subject.send($0)
            }
    }
}

private extension DefaultAlbumsProvider {

    func fetchAlbums(type: PHAssetCollectionType) -> [AlbumModel] {
        let options = PHFetchOptions()
        options.includeAssetSourceTypes = [.typeUserLibrary, .typeiTunesSynced, .typeCloudShared]
        options.sortDescriptors = [NSSortDescriptor(key: "localizedTitle", ascending: true)]

        let collections = PHAssetCollection.fetchAssetCollections(
            with: type,
            subtype: .any,
            options: options
        )

        if collections.count == 0 {
            return []
        }
        var albums: [AlbumModel] = []

        for index in 0...(collections.count - 1) {
            let collection = collections[index]
            let options = PHFetchOptions()
            options.sortDescriptors = [
                NSSortDescriptor(key: "creationDate", ascending: false)
            ]
            options.fetchLimit = 1
            let fetchResult = PHAsset.fetchAssets(in: collection, options: options)
            if fetchResult.count == 0 {
                continue
            }
            let preview = MediasProvider.map(fetchResult: fetchResult, mediaSelectionType: mediaSelectionType).first
            let album = AlbumModel(preview: preview, source: collection)
            albums.append(album)
        }
        return albums
    }
}
