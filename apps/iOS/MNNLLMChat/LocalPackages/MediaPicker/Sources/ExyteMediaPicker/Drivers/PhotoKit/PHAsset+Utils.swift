//
//  Created by Alex.M on 31.05.2022.
//

import Foundation
import Photos
import UniformTypeIdentifiers

#if os(iOS)
import UIKit.UIImage
import UIKit.UIScreen
#endif

extension PHAsset {
    actor RequestStore {
        var request: Request?
        
        func storeRequest(_ request: Request) {
            self.request = request
        }
        
        func cancel(asset: PHAsset) {
            switch request {
            case .contentEditing(let id):
                asset.cancelContentEditingInputRequest(id)
            case .imageRequest(let id):
                PHCachingImageManager.default().cancelImageRequest(id)
            case .none:
                break
            }
        }
    }
    
    enum Request {
        case contentEditing(PHContentEditingInputRequestID)
        case imageRequest(PHImageRequestID)
    }
    
    func getURLCancellableRequest(completion: @escaping (URL?) -> Void) -> Request? {
        var request: Request?
        
        if mediaType == .image {
            let options = PHContentEditingInputRequestOptions()
            options.isNetworkAccessAllowed = true
            options.canHandleAdjustmentData = { _ -> Bool in
                return true
            }
            request = .contentEditing(
                requestContentEditingInput(
                    with: options,
                    completionHandler: { (contentEditingInput, _) in
                        completion(contentEditingInput?.fullSizeImageURL)
                    }
                )
            )
        } else if mediaType == .video {
            let options = PHVideoRequestOptions()
            options.version = .current
            options.isNetworkAccessAllowed = true
            options.deliveryMode = .highQualityFormat

            request = .imageRequest(
                PHCachingImageManager.default().requestAVAsset(forVideo: self, options: options) { avAsset, audio, info in
                    let asset = avAsset as? AVURLAsset
                    completion(asset?.url)
                }
            )
        }
        
        return request
    }

    var formattedDuration: String? {
        guard mediaType == .video || mediaType == .audio else {
            return nil
        }
        return duration.formatted()
    }
}

extension PHAsset {
    func getURL() async -> URL? {
        let requestStore = RequestStore()
        
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let request = getURLCancellableRequest { url in
                    continuation.resume(returning: url)
                }
                if let request = request {
                    Task {
                        await requestStore.storeRequest(request)
                    }
                }
            }
        } onCancel: {
            Task {
                await requestStore.cancel(asset: self)
            }
        }
    }

    func getThumbnailURL() async -> URL? {
        guard let url = await getURL() else { return nil }
        if mediaType == .image {
            return url
        }

        let asset: AVAsset = AVAsset(url: url)
        if let thumbnailData = asset.generateThumbnail() {
            return FileManager.storeToTempDir(data: thumbnailData)
        }

        return nil
    }

    func getThumbnailData() async -> Data? {
        if mediaType == .image {
            return try? await self.getData()
        }
        else if mediaType == .video {
            guard let url = await getURL() else { return nil }
            let asset: AVAsset = AVAsset(url: url)
            return asset.generateThumbnail()
        }
        return nil
    }
}

extension CGImage {
    var jpegData: Data? {
        guard let mutableData = CFDataCreateMutable(nil, 0),
              let destination = CGImageDestinationCreateWithData(mutableData, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(destination, self, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return mutableData as Data
    }
}

#if os(iOS)
extension PHAsset {

    func image(size: CGSize, resultClosure: @escaping (UIImage?)->()) -> PHImageRequestID {
        let requestSize = CGSize(width: size.width * UIScreen.main.scale, height: size.height * UIScreen.main.scale)

        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = true
        options.deliveryMode = .opportunistic

        return PHCachingImageManager.default().requestImage(
            for: self,
            targetSize: requestSize,
            contentMode: .aspectFill,
            options: options,
            resultHandler: { image, info in
                resultClosure(image) // called for every quality approximation
            }
        )
    }

    func getData() async throws -> Data? {
        try await withCheckedThrowingContinuation { continuation in
            if mediaType == .image {
                let options = PHImageRequestOptions()
                options.isNetworkAccessAllowed = true
                options.deliveryMode = .highQualityFormat
                options.isSynchronous = true

                PHCachingImageManager.default().requestImageDataAndOrientation(
                    for: self,
                    options: options,
                    resultHandler: { data, _, _, info in
                        guard info?.keys.contains(PHImageResultIsDegradedKey) == true
                        else { fatalError("PHImageManager with `options.isSynchronous = true` should call result ONE time.") }
                        if let data = data {
                            continuation.resume(returning: data)
                        } else {
                            continuation.resume(throwing: AssetFetchError.noImageData)
                        }
                    }
                )
            }
            else if mediaType == .video {
                let options = PHVideoRequestOptions()
                options.isNetworkAccessAllowed = true
                options.deliveryMode = .highQualityFormat
                options.version = .current

                PHCachingImageManager.default().requestAVAsset(forVideo: self, options: options) { avAsset, audio, info in
                    do {
                        if let asset = avAsset as? AVURLAsset {
                            let data = try Data(contentsOf: asset.url)
                            continuation.resume(returning: data)
                        }
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            } else {
                continuation.resume(throwing: AssetFetchError.unknownType)
            }
        }
    }
}

extension AVAsset {
    func generateThumbnail() -> Data? {
        let imageGenerator = AVAssetImageGenerator(asset: self)
        imageGenerator.appliesPreferredTrackTransform = true
        do {
            let thumbnailImage = try imageGenerator.copyCGImage(at: CMTimeMake(value: 1, timescale: 60), actualTime: nil)
            guard let data = thumbnailImage.jpegData else { return nil }
            return data
        } catch let error {
            print(error)
        }
        return nil
    }
}

enum AssetFetchError: Error {
    case noImageData
    case unknownType
}
#endif
