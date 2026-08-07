//
//  Created by Alex.M on 16.06.2022.
//

import SwiftUI

struct AttachmentCell: View {

    @Environment(\.chatTheme) private var theme

    let attachment: Attachment
    let onTap: (Attachment) -> Void

    @StateObject private var imageSaver = ImageSaver()
    @State private var showSaveDialog = false

    var body: some View {
        Group {
            if attachment.type == .image {
                content
            } else if attachment.type == .video {
                content
                    .overlay {
                        theme.images.message.playVideo
                            .resizable()
                            .foregroundColor(.white)
                            .frame(width: 36, height: 36)
                    }
            } else {
                content
                    .overlay {
                        Text("Unknown", bundle: .module)
                    }
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            onTap(attachment)
        }
        .applyIf(attachment.type == .image) { view in
            view.onLongPressGesture {
                showSaveDialog = true
            }
        }
        .confirmationDialog(
            NSLocalizedString("Save to Photos", comment: "Save image dialog title"),
            isPresented: $showSaveDialog,
            titleVisibility: .visible
        ) {
            Button(NSLocalizedString("Save to Photos", comment: "Save image to photo library")) {
                imageSaver.saveToPhotoLibrary(url: attachment.full)
            }
            Button(NSLocalizedString("Cancel", comment: "Cancel action"), role: .cancel) {}
        }
        .alert(imageSaver.alertMessage, isPresented: $imageSaver.showAlert) {
            Button("OK", role: .cancel) {}
        }
    }

    var content: some View {
        AsyncImageView(url: attachment.thumbnail)
    }
}

struct AsyncImageView: View {

    @Environment(\.chatTheme) var theme
    let url: URL

    var body: some View {
        CachedAsyncImage(url: url, urlCache: .imageCache) { imageView in
            imageView
                .resizable()
                .scaledToFill()
        } placeholder: {
            ZStack {
                Rectangle()
                    .foregroundColor(theme.colors.inputBG)
                    .frame(minWidth: 100, minHeight: 100)
                ActivityIndicator(size: 30, showBackground: false)
            }
        }
    }
}
