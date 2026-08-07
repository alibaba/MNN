//
//  Created by Alex.M on 22.06.2022.
//

import Foundation
import SwiftUI

struct FullscreenMediaPages: View {

    @Environment(\.chatTheme) private var theme
    @Environment(\.mediaPickerTheme) var pickerTheme

    @StateObject var viewModel: FullscreenMediaPagesViewModel
    @StateObject private var imageSaver = ImageSaver()
    @State private var showSaveDialog = false
    var safeAreaInsets: EdgeInsets
    var onClose: () -> Void

    var body: some View {
        let closeGesture = DragGesture()
            .onChanged { viewModel.offset = closeSize(from: $0.translation) }
            .onEnded {
                withAnimation {
                    viewModel.offset = .zero
                }
                if $0.translation.height >= 100 {
                    onClose()
                }
            }

        ZStack {
            Color.black
                .opacity(max((200.0 - viewModel.offset.height) / 200.0, 0.5))

            VStack {
                HStack {
                    Spacer()
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 32))
                            .foregroundColor(.white)
                            .shadow(radius: 4)
                    }
                    .padding(.trailing, 20)
                }
                .padding(.top, safeAreaInsets.top + 8)
                Spacer()
            }
            .zIndex(10)
            VStack {
                TabView(selection: $viewModel.index) {
                    ForEach(viewModel.attachments.enumerated().map({ $0 }), id: \.offset) { (index, attachment) in
                        AttachmentsPage(attachment: attachment)
                            .tag(index)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .allowsHitTesting(false)
                            .ignoresSafeArea()
                    }
                    .ignoresSafeArea()
                }
                .environmentObject(viewModel)
                .tabViewStyle(.page(indexDisplayMode: .never))
            }
            .offset(viewModel.offset)
            .gesture(closeGesture)
            .onTapGesture {
                withAnimation {
                    viewModel.showMinis.toggle()
                }
            }
            .onLongPressGesture {
                let current = viewModel.attachments[viewModel.index]
                if current.type == .image {
                    showSaveDialog = true
                }
            }

            VStack {
                Spacer()
                ScrollViewReader { proxy in
                    if viewModel.showMinis {
                        ScrollView(.horizontal) {
                            HStack(spacing: 2) {
                                ForEach(viewModel.attachments.enumerated().map({ $0 }), id: \.offset) { (index, attachment) in
                                    AttachmentCell(attachment: attachment) { _ in
                                        withAnimation {
                                            viewModel.index = index
                                        }
                                    }
                                    .frame(width: 100, height: 100)
                                    .cornerRadius(4)
                                    .clipped()
                                    .id(index)
                                    .overlay {
                                        if viewModel.index == index {
                                            RoundedRectangle(cornerRadius: 4)
                                                .stroke(theme.colors.sendButtonBackground, lineWidth: 2)
                                        }
                                    }
                                    .padding(.vertical, 1)
                                }
                            }
                        }
                        .padding([.top, .horizontal], 12)
                        .background(Color.black)
                        .onAppear {
                            proxy.scrollTo(viewModel.index)
                        }
                        .onChange(of: viewModel.index) { newValue in
                            withAnimation {
                                proxy.scrollTo(newValue, anchor: .center)
                            }
                        }
                    }
                }
                .offset(y: -safeAreaInsets.bottom)
            }
            .offset(viewModel.offset)
        }
        .ignoresSafeArea()
        .overlay(alignment: .top) {
            if viewModel.showMinis {
                Text("\(viewModel.index + 1)/\(viewModel.attachments.count)")
                    .foregroundColor(.white)
                    .offset(y: safeAreaInsets.top)
            }
        }
        .overlay(alignment: .topLeading) {
            if viewModel.showMinis {
                Button(action: onClose) {
                    theme.images.mediaPicker.cross
                        .imageScale(.large)
                        .padding(5)
                }
                .tint(.white)
                .padding(.leading, 15)
                .offset(y: safeAreaInsets.top - 5)
            }
        }
        .overlay(alignment: .topTrailing) {
            if viewModel.showMinis {
                HStack(spacing: 20) {
                    if viewModel.attachments[viewModel.index].type == .image {
                        Image(systemName: "square.and.arrow.down")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 22, height: 22)
                            .padding(5)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                showSaveDialog = true
                            }
                    }

                    if viewModel.attachments[viewModel.index].type == .video {
                        (viewModel.videoPlaying ? theme.images.fullscreenMedia.pause : theme.images.fullscreenMedia.play)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 24, height: 24)
                            .padding(5)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                viewModel.toggleVideoPlaying()
                            }

                        (viewModel.videoMuted ? theme.images.fullscreenMedia.unmute : theme.images.fullscreenMedia.mute)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 24, height: 24)
                            .padding(5)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                viewModel.toggleVideoMuted()
                            }
                    }
                }
                .foregroundColor(.white)
                .padding(.trailing, 10)
                .offset(y: safeAreaInsets.top - 5)
            }
        }
        .confirmationDialog(
            NSLocalizedString("Save to Photos", comment: "Save image dialog title"),
            isPresented: $showSaveDialog,
            titleVisibility: .visible
        ) {
            Button(NSLocalizedString("Save to Photos", comment: "Save image to photo library")) {
                let current = viewModel.attachments[viewModel.index]
                imageSaver.saveToPhotoLibrary(url: current.full)
            }
            Button(NSLocalizedString("Cancel", comment: "Cancel action"), role: .cancel) {}
        }
        .alert(imageSaver.alertMessage, isPresented: $imageSaver.showAlert) {
            Button("OK", role: .cancel) {}
        }
    }
}

private extension FullscreenMediaPages {
    func closeSize(from size: CGSize) -> CGSize {
        CGSize(width: 0, height: max(size.height, 0))
    }
}
