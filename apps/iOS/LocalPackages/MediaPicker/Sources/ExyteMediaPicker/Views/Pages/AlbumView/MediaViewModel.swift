//
//  Created by Alex.M on 03.06.2022.
//

#if os(iOS)
import UIKit.UIImage
#endif
import Photos

class MediaViewModel: ObservableObject {
    let assetMediaModel: AssetMediaModel
    
    private var requestID: PHImageRequestID?
    
    init(assetMediaModel: AssetMediaModel) {
        self.assetMediaModel = assetMediaModel
    }
    
#if os(iOS)
    @Published var preview: UIImage? = nil
#else
    // FIXME: Create preview for image/video for other platforms
#endif
    
    func onStart(size: CGSize) {
        requestID = assetMediaModel.asset
            .image(size: size) {
                self.preview = $0
            }
    }
    
    func onStop() {
        if let requestID = requestID {
            PHCachingImageManager.default().cancelImageRequest(requestID)
        }
    }
}
