//
//  Created by Alex.M on 27.05.2022.
//

import SwiftUI
import Combine

struct AlbumsView: View {

    @EnvironmentObject private var selectionService: SelectionService
    @EnvironmentObject private var permissionsService: PermissionsService
    @Environment(\.mediaPickerTheme) private var theme

    @StateObject var viewModel: AlbumsViewModel
    @ObservedObject var mediaPickerViewModel: MediaPickerViewModel

    @Binding var showingCamera: Bool
    @Binding var currentFullscreenMedia: Media?

    let selectionParamsHolder: SelectionParamsHolder
    let filterClosure: MediaPicker.FilterClosure?
    let massFilterClosure: MediaPicker.MassFilterClosure?

    @State private var showingLoadingCell = false

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: 100), spacing: 0, alignment: .top)]
    }
    
    private var cellPadding: EdgeInsets {
        EdgeInsets(top: 2, leading: 2, bottom: 8, trailing: 2)
    }
    
    var body: some View {
        ScrollView {
            VStack {
                if let action = permissionsService.photoLibraryAction {
                    PermissionsActionView(action: .library(action))
                }
                if viewModel.isLoading {
                    ProgressView()
                        .padding()
                } else if viewModel.albums.isEmpty {
                    Text("Empty data")
                        .font(.title3)
                        .foregroundColor(theme.main.text)
                } else {
                    LazyVGrid(columns: columns, spacing: 0) {
                        ForEach(viewModel.albums) { album in
                            AlbumCell(
                                viewModel: AlbumCellViewModel(album: album)
                            )
                            .padding(cellPadding)
                            .onTapGesture {
                                mediaPickerViewModel.setPickerMode(.album(album.toAlbum()))
                            }
                        }
                    }
                }
                Spacer()
            }
        }
        .onAppear {
            viewModel.onStart()
        }
        .onDisappear {
            viewModel.onStop()
        }
    }
}
