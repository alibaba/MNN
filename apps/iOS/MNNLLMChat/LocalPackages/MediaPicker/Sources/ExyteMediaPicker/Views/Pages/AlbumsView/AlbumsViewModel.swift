//
//  Created by Alex.M on 07.06.2022.
//

import Foundation
import Combine

final class AlbumsViewModel: ObservableObject {
    // MARK: - Values
    // MARK: Public
    @Published var albums: [AlbumModel] = []
    @Published var isLoading: Bool = false
    
    let albumsProvider: AlbumsProviderProtocol

    // MARK: Private
    private var albumsCancellable: AnyCancellable?
    
    // MARK: - Object life cycle
    init(albumsProvider: AlbumsProviderProtocol) {
        self.albumsProvider = albumsProvider
    }
    
    // MARK: - Public methods
    func onStart() {
        isLoading = true
        albumsCancellable = albumsProvider.albums
            .receive(on: DispatchQueue.main)
            .sink { [weak self] in
                self?.albums = $0
                self?.isLoading = false
            }
        
        albumsProvider.reload()
    }
    
    func onStop() {
        albumsCancellable = nil
    }
}
