//
//  Created by Alex.M on 03.06.2022.
//

#if os(iOS)
import UIKit.UIImage
#endif
import Photos

class AlbumCellViewModel: ObservableObject {
    let album: AlbumModel

    private var requestID: PHImageRequestID?
    
    init(album: AlbumModel) {
        self.album = album
    }
    
#if os(iOS)
    @Published var preview: UIImage? = nil
#else
    // FIXME: Create preview for image/video for other platforms
#endif
    
    func fetchPreview(size: CGSize) {
        guard preview == nil else { return }
        
        requestID = album.preview?.asset
            .image(size: size) { [weak self] in
                self?.preview = $0
            }
    }

    func onStop() {
        if let requestID = requestID {
            PHCachingImageManager.default().cancelImageRequest(requestID)
        }
    }
}
