//
//  UploadingManager.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 26.06.2023.
//

import Foundation
import FirebaseStorage

class UploadingManager {

    static func uploadImageMedia(_ media: Media?) async -> URL? {
        guard let data = await media?.getData() else { return nil }
        let ref = Storage.storage().reference()
            .child("\(UUID().uuidString).jpg")
        return await uploadData(data, ref)
    }

    static func uploadVideoMedia(_ media: Media?) async -> (URL?, URL?) { // thumbnailURL, fullURL
        guard let thumbData = await media?.getThumbnailData(), let data = await media?.getData() else { return (nil, nil) }
        let thumbRef = Storage.storage().reference()
            .child("\(UUID().uuidString).jpg")
        let ref = Storage.storage().reference()
            .child("\(UUID().uuidString).mov")
        return (await uploadData(thumbData, thumbRef), await uploadData(data, ref))
    }

    static func uploadRecording(_ recording: Recording?) async -> URL? {
        guard let url = recording?.url, let data = try? Data(contentsOf: url) else { return nil }
        let ref = Storage.storage().reference()
            .child("\(UUID().uuidString).aac")
        return await uploadData(data, ref)
    }

    static func uploadImageData(_ data: Data?) async -> URL? {
        guard let data = data else { return nil }
        let ref = Storage.storage().reference()
            .child("\(UUID().uuidString).jpg")
        return await uploadData(data, ref)
    }

    static private func uploadData(_ data: Data, _ ref: StorageReference) async -> URL? {
        await withCheckedContinuation { continuation in
            ref.putData(data, metadata: nil) { metadata, error in
                guard let _ = metadata else {
                    print(error.debugDescription)
                    continuation.resume(returning: nil)
                    return
                }
                ref.downloadURL { (url, error) in
                    guard let downloadURL = url else {
                        print(error.debugDescription)
                        continuation.resume(returning: nil)
                        return
                    }
                    continuation.resume(returning: downloadURL)
                }
            }
        }
    }
}
