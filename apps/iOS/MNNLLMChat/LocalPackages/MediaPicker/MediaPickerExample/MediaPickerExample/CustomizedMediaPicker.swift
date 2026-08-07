//
//  Created by Alex.M on 05.07.2022.
//

import Foundation
import SwiftUI
import ExyteMediaPicker

struct CustomizedMediaPicker: View {

    @EnvironmentObject private var appDelegate: AppDelegate

    @Binding var isPresented: Bool
    @Binding var medias: [Media]

    @State private var albums: [Album] = []

    @State private var mediaPickerMode = MediaPickerMode.photos
    @State private var selectedAlbum: Album?
    @State private var currentFullscreenMedia: Media?
    @State private var showAlbumsDropDown: Bool = false
    @State private var videoIsBeingRecorded: Bool = false

    let maxCount: Int = 5

    var body: some View {
        MediaPicker(
            isPresented: $isPresented,
            onChange: { medias = $0 },
            albumSelectionBuilder: { _, albumSelectionView, _ in
                VStack {
                    headerView
                    albumSelectionView
                    Spacer()
                    footerView
                        .background(Color.black)
                }
                .background(Color.black)
            },
            cameraSelectionBuilder: { addMoreClosure, cancelClosure, cameraSelectionView in
                VStack {
                    HStack {
                        Spacer()
                        Button("Done", action: { isPresented = false })
                    }
                    .padding()
                    cameraSelectionView
                    HStack {
                        Button("Cancel", action: cancelClosure)
                        Spacer()
                        Button(action: addMoreClosure) {
                            Text("Take more photos")
                                .greenButtonStyle()
                        }
                    }
                    .padding()
                }
                .background(Color.black)
            },
            cameraViewBuilder: { cameraSheetView, cancelClosure, showPreviewClosure, takePhotoClosure, startVideoCaptureClosure, stopVideoCaptureClosure, _, _ in
                cameraSheetView
                    .overlay(alignment: .topLeading) {
                        HStack {
                            Button("Cancel") { cancelClosure() }
                                .foregroundColor(Color("CustomPurple"))
                            Spacer()
                            Button("Done") { showPreviewClosure() }
                                .foregroundColor(Color("CustomPurple"))
                        }
                        .padding()
                    }
                    .overlay(alignment: .bottom) {
                        HStack {
                            Button("Take photo") { takePhotoClosure() }
                                .greenButtonStyle()
                            Button(videoIsBeingRecorded ? "Stop video capture" : "Capture video") {
                                videoIsBeingRecorded ? stopVideoCaptureClosure() : startVideoCaptureClosure()
                                videoIsBeingRecorded.toggle()
                            }
                            .greenButtonStyle()
                        }
                        .padding()
                    }
            }
        )
        .showLiveCameraCell()
        .albums($albums)
        .pickerMode($mediaPickerMode)
        .currentFullscreenMedia($currentFullscreenMedia)
        .orientationHandler {
            switch $0 {
            case .lock: appDelegate.lockOrientationToPortrait()
            case .unlock: appDelegate.unlockOrientation()
            }
        }
        .mediaSelectionStyle(.count)
        .mediaSelectionLimit(maxCount)
        .mediaPickerTheme(
            main: .init(
                albumSelectionBackground: .black,
                fullscreenPhotoBackground: .black
            ),
            selection: .init(
                emptyTint: .white,
                emptyBackground: .black.opacity(0.25),
                selectedTint: Color("CustomPurple"),
                fullscreenTint: .white
            )
        )
        .overlay(alignment: .topLeading) {
            if showAlbumsDropDown {
                albumsDropdown
                    .background(Color.white)
                    .foregroundColor(.black)
                    .cornerRadius(5)
            }
        }
        .background(Color.black)
        .foregroundColor(.white)
    }

    var headerView: some View {
        HStack {
            HStack {
                Text(selectedAlbum?.title ?? "Recents")
                Image(systemName: "chevron.down")
                    .rotationEffect(Angle(radians: showAlbumsDropDown ? .pi : 0))
            }
            .onTapGesture {
                withAnimation {
                    showAlbumsDropDown.toggle()
                }
            }

            Spacer()

            Text("\(medias.count) out of \(maxCount) selected")
        }
        .padding()
    }

    var footerView: some View {
        HStack {
            TextField("", text: .constant(""), prompt: Text("Add a caption").foregroundColor(.gray))
                .padding()
                .background {
                    Color.white.opacity(0.2)
                        .cornerRadius(6)
                }

            Spacer(minLength: 70)

            Button {
                isPresented = false
            } label: {
                HStack {
                    Text("Add")

                    Text("\(medias.count)")
                        .padding(6)
                        .background(Color.white)
                        .clipShape(Circle())
                }
                .frame(maxWidth: .infinity)
            }
            .greenButtonStyle()
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }

    var albumsDropdown: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(albums) { album in
                    Button(album.title ?? "") {
                        selectedAlbum = album
                        mediaPickerMode = .album(album)
                        showAlbumsDropDown = false
                    }
                }
            }
            .padding(15)
        }
        .frame(maxHeight: 300)
    }
}

extension View {
    func greenButtonStyle() -> some View {
        self.font(.headline)
            .foregroundColor(.black)
            .padding()
            .background {
                Color("CustomGreen")
                    .cornerRadius(16)
            }
    }
}
