//
//  AvatarView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 19.06.2023.
//

import SwiftUI
import CachedAsyncImage

struct AvatarView: View {
    let url: URL?
    let size: CGFloat

    var body: some View {
        if let url {
            CachedAsyncImage(url: url, urlCache: .imageCache) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                Image(.placeholderAvatar)
                    .resizable()
            }
            .frame(width: size, height: size)
            .clipShape(Circle())
        } else {
            Image(.placeholderAvatar)
                .resizable()
                .frame(width: size, height: size)
        }
    }
}

extension URLCache {
    static let imageCache = URLCache(
        memoryCapacity: 512.megabytes(),
        diskCapacity: 2.gigabytes()
    )
}

private extension Int {
    func kilobytes() -> Int {
        self * 1024 * 1024
    }

    func megabytes() -> Int {
        self.kilobytes() * 1024
    }

    func gigabytes() -> Int {
        self.megabytes() * 1024
    }
}
