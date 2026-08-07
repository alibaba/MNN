//
//  Created by Alex.M on 09.06.2022.
//

import Foundation
import Combine

final class AlbumViewModel: ObservableObject {

    @Published var title: String? = nil
    @Published var assetMediaModels: [AssetMediaModel] = []
    @Published var isLoading: Bool = false
    
    let mediasProvider: MediasProviderProtocol

    private var mediaCancellable: AnyCancellable?
    
    init(mediasProvider: MediasProviderProtocol) {
        self.mediasProvider = mediasProvider
        onStart()
    }
    
    func onStart() {
        isLoading = true
        mediaCancellable = mediasProvider.assetMediaModelsPublisher
            .receive(on: RunLoop.main)
            .sink { [weak self] in
                self?.assetMediaModels = $0
                self?.isLoading = false
            }
        
        mediasProvider.reload()
    }

    deinit {
        mediasProvider.cancel()
        mediaCancellable = nil
    }
}
