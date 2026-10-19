//
//  AttachmentsEditor.swift
//  Chat
//
//  Created by Alex.M on 22.06.2022.
//

import ActivityIndicatorView
import ExyteMediaPicker
import SwiftUI

struct AttachmentsEditor<InputViewContent: View>: View {
    typealias InputViewBuilderClosure = ChatView<EmptyView, InputViewContent, DefaultMessageMenuAction>.InputViewBuilderClosure

    @Environment(\.chatTheme) var theme
    @Environment(\.mediaPickerTheme) var pickerTheme

    @EnvironmentObject private var keyboardState: KeyboardState
    @EnvironmentObject private var globalFocusState: GlobalFocusState

    @ObservedObject var inputViewModel: InputViewModel

    var inputViewBuilder: InputViewBuilderClosure?
    var chatTitle: String?
    var messageUseMarkdown: Bool
    var orientationHandler: MediaPickerOrientationHandler
    var mediaPickerSelectionParameters: MediaPickerParameters?
    var availableInput: AvailableInputType
    var localization: ChatLocalization

    @State private var seleсtedMedias: [Media] = []
    @State private var currentFullscreenMedia: Media?
    @State private var originalMedias: [Media] = []

    var showingAlbums: Bool {
        inputViewModel.mediaPickerMode == .albums
    }

    var body: some View {
        ZStack {
            mediaPicker

            if inputViewModel.showActivityIndicator {
                ActivityIndicator()
            }
        }
        .onAppear { originalMedias = inputViewModel.attachments.medias }
    }

    var mediaPicker: some View {
        GeometryReader { g in
            MediaPicker(isPresented: $inputViewModel.showPicker) {
                seleсtedMedias = $0
                assembleSelectedMedia()
            } albumSelectionBuilder: { _, albumSelectionView, _ in
                VStack {
                    albumSelectionHeaderView
                        .padding(.top, g.safeAreaInsets.top)
                    albumSelectionView
                    Spacer()
                    inputView
                        .padding(.bottom, g.safeAreaInsets.bottom)
                }
                .background(theme.colors.mainBG)
                .ignoresSafeArea()
            } cameraSelectionBuilder: { _, _, cameraSelectionView in
                VStack {
                    cameraSelectionHeaderView(cancelClosure: cancelSelection)
                        .padding(.top, g.safeAreaInsets.top)
                    cameraSelectionView
                    Spacer()
                    Button {
                        inputViewModel.showPicker = false
                    } label: {
                        Text(LocalizedStringKey("chat.attachments.usePhoto"))
                            .frame(maxWidth: .infinity).padding()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(inputViewModel.attachments.medias.isEmpty)
                    .padding(.horizontal)
                    .padding(.bottom, max(g.safeAreaInsets.bottom, 12))
                }
                .ignoresSafeArea()
            }
            .didPressCancelCamera {
                cancelSelection()
            }
            .currentFullscreenMedia($currentFullscreenMedia)
            .setSelectionParameters(mediaPickerSelectionParameters)
            .pickerMode($inputViewModel.mediaPickerMode)
            .orientationHandler(orientationHandler)
            .padding(.top)
            .background(theme.colors.mainBG)
            .ignoresSafeArea(.all)
            .onChange(of: currentFullscreenMedia) { newValue in
                if newValue != nil { assembleSelectedMedia() }
            }
            .mediaPickerTheme(
                main: .init(
                    text: theme.colors.mainText,
                    albumSelectionBackground: theme.colors.mainBG,
                    fullscreenPhotoBackground: theme.colors.mainBG,
                    cameraBackground: theme.colors.mainBG,
                    cameraSelectionBackground: theme.colors.mainBG
                ),
                selection: .init(
                    selectedTint: theme.colors.sendButtonBackground,
                    fullscreenTint: theme.colors.sendButtonBackground
                )
            )
        }
    }

    func cancelSelection() {
        inputViewModel.attachments.medias = originalMedias
        inputViewModel.showPicker = false
    }

    func assembleSelectedMedia() {
        if !seleсtedMedias.isEmpty {
            inputViewModel.attachments.medias = seleсtedMedias
        } else if let media = currentFullscreenMedia {
            inputViewModel.attachments.medias = [media]
        } else {
            inputViewModel.attachments.medias = []
        }
    }

    @ViewBuilder
    var inputView: some View {
        Group {
            if let inputViewBuilder = inputViewBuilder {
                inputViewBuilder($inputViewModel.text, inputViewModel.attachments, inputViewModel.state, .signature, inputViewModel.inputViewAction()) {
                    globalFocusState.focus = nil
                }
            } else {
                InputView(
                    viewModel: inputViewModel,
                    inputFieldId: UUID(),
                    style: .signature,
                    availableInput: availableInput,
                    messageUseMarkdown: messageUseMarkdown,
                    localization: localization
                )
            }
        }
    }

    var albumSelectionHeaderView: some View {
        ZStack {
            HStack {
                Button {
                    cancelSelection()
                } label: {
                    Text(localization.cancelButtonText)
                }

                Spacer()
            }

            HStack {
                Text(localization.recentToggleText)
                Image(systemName: "chevron.down")
                    .rotationEffect(Angle(radians: showingAlbums ? .pi : 0))
            }
            .onTapGesture {
                withAnimation {
                    inputViewModel.mediaPickerMode = showingAlbums ? .photos : .albums
                }
            }
            .frame(maxWidth: .infinity)
        }
        .foregroundColor(theme.colors.mainText)
        .padding(.horizontal)
        .padding(.bottom, 5)
    }

    func cameraSelectionHeaderView(cancelClosure: @escaping () -> Void) -> some View {
        HStack {
            Button {
                cancelClosure()
            } label: {
                theme.images.mediaPicker.cross
            }
            .padding(.trailing, 30)

            if let chatTitle = chatTitle {
                theme.images.mediaPicker.chevronRight
                Text(chatTitle)
                    .font(.title3)
                    .foregroundColor(theme.colors.mainText)
            }

            Spacer()
        }
        .padding(.horizontal)
        .padding(.bottom, 10)
    }
}
