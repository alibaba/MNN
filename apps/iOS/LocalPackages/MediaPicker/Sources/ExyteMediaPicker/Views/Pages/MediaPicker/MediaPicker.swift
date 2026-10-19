//
//  Created by Alex.M on 26.05.2022.
//

import SwiftUI
import Combine

public struct MediaPicker<AlbumSelectionContent: View, CameraSelectionContent: View, CameraViewContent: View>: View {

    /// To provide custom buttons layout for photos grid view use actions and views provided by this closure:
    /// - standard header with photos/albums switcher
    /// - selection view you can embed in your view
    /// - is in fullscreen photo details mode
    public typealias AlbumSelectionClosure = ((ModeSwitcher, AlbumSelectionView, Bool) -> AlbumSelectionContent)

    /// To provide custom buttons layout for camera selection view use actions and views provided by this closure:
    /// - add more photos closure
    /// - cancel closure
    /// - selection view you can embed in your view
    public typealias CameraSelectionClosure = ((@escaping SimpleClosure, @escaping SimpleClosure, CameraSelectionView) -> CameraSelectionContent)

    /// To provide custom buttons layout for camera  view use actions and views provided by this closure:
    /// - live camera capture view
    /// - cancel closure
    /// - show preview of taken photos
    /// - take photo closure
    /// - start record video closure
    /// - stop record video closure
    /// - flash off/on closure
    /// - camera back/front closure
    public typealias CameraViewClosure = ((LiveCameraView, @escaping SimpleClosure, @escaping SimpleClosure, @escaping SimpleClosure, @escaping SimpleClosure, @escaping SimpleClosure, @escaping SimpleClosure, @escaping SimpleClosure) -> CameraViewContent)

    public typealias FilterClosure = (Media) async -> Media?
    public typealias MassFilterClosure = ([Media]) async -> [Media]

    // MARK: - Parameters

    @Binding private var isPresented: Bool
    private let onChange: MediaPickerCompletionClosure

    // MARK: - View builders

    private var albumSelectionBuilder: AlbumSelectionClosure? = nil
    private var cameraSelectionBuilder: CameraSelectionClosure? = nil
    private var cameraViewBuilder: CameraViewClosure? = nil

    // MARK: - Customization

    @Binding private var albums: [Album]
    @Binding private var currentFullscreenMediaBinding: Media?

    private var pickerMode: Binding<MediaPickerMode>?
    private var showingLiveCameraCell: Bool = false
    private var didPressCancelCamera: (() -> Void)?
    private var orientationHandler: MediaPickerOrientationHandler = {_ in}
    private var filterClosure: FilterClosure?
    private var massFilterClosure: MassFilterClosure?
    private var selectionParamsHolder = SelectionParamsHolder()

    // MARK: - Inner values

    @Environment(\.mediaPickerTheme) private var theme

    @StateObject private var viewModel = MediaPickerViewModel()
    @StateObject private var selectionService = SelectionService()
    @StateObject private var cameraSelectionService = CameraSelectionService()
    @StateObject private var permissionService = PermissionsService()

    @State private var readyToShowCamera = false
    @State private var currentFullscreenMedia: Media?

    @State private var internalPickerMode: MediaPickerMode = .photos // a hack for slow camera dismissal

    var isInFullscreen: Bool {
        currentFullscreenMedia != nil
    }

    // MARK: - Object life cycle

    public init(isPresented: Binding<Bool>,
                onChange: @escaping MediaPickerCompletionClosure,
                albumSelectionBuilder: AlbumSelectionClosure? = nil,
                cameraSelectionBuilder: CameraSelectionClosure? = nil,
                cameraViewBuilder: CameraViewClosure? = nil) {

        self._isPresented = isPresented
        self._albums = .constant([])
        self._currentFullscreenMediaBinding = .constant(nil)

        self.onChange = onChange
        self.albumSelectionBuilder = albumSelectionBuilder
        self.cameraSelectionBuilder = cameraSelectionBuilder
        self.cameraViewBuilder = cameraViewBuilder
    }

    public var body: some View {
        Group {
            switch internalPickerMode { // please don't use viewModel.internalPickerMode here - it slows down camera dismissal
                case .photos, .albums, .album(_):
                    albumSelectionContainer
                case .camera:
                    cameraContainer
                case .cameraSelection:
                    cameraSelectionContainer
                }
        }
        .background(theme.main.albumSelectionBackground.ignoresSafeArea())
        .environmentObject(selectionService)
        .environmentObject(cameraSelectionService)
        .environmentObject(permissionService)
        .onAppear {
            if showingLiveCameraCell {
                permissionService.requestCameraPermission()
            }
            permissionService.checkPhotoLibraryAuthorizationStatus()

            selectionService.onChange = onChange
            selectionService.mediaSelectionLimit = selectionParamsHolder.selectionLimit
            
            cameraSelectionService.onChange = onChange
            cameraSelectionService.mediaSelectionLimit = selectionParamsHolder.selectionLimit

            viewModel.shouldUpdatePickerMode = { mode in
                pickerMode?.wrappedValue = mode
            }
            viewModel.onStart()
        }
        .onChange(of: viewModel.albums) {
            self.albums = $0.map { $0.toAlbum() }
        }
        .onChange(of: pickerMode?.wrappedValue) { mode in
            if let mode = mode {
                viewModel.setPickerMode(mode)
            }
        }
        .onChange(of: viewModel.internalPickerMode) { newValue in
            internalPickerMode = newValue
        }
        .onChange(of: currentFullscreenMedia) { currentFullscreenMedia in
            _currentFullscreenMediaBinding.wrappedValue = currentFullscreenMedia
        }
        .onAppear {
            if let mode = pickerMode?.wrappedValue {
                viewModel.setPickerMode(mode)
            }
        }
    }

    @ViewBuilder
    var albumSelectionContainer: some View {
        let albumSelectionView = AlbumSelectionView(viewModel: viewModel, showingCamera: cameraBinding(), currentFullscreenMedia: $currentFullscreenMedia, showingLiveCameraCell: showingLiveCameraCell, selectionParamsHolder: selectionParamsHolder, filterClosure: filterClosure, massFilterClosure: massFilterClosure) {
            // has media limit of 1, and it's been selected
            isPresented = false
        }

        if let albumSelectionBuilder = albumSelectionBuilder {
            albumSelectionBuilder(ModeSwitcher(selection: modeBinding()), albumSelectionView, isInFullscreen)
        } else {
            VStack(spacing: 0) {
                if !isInFullscreen {
                    defaultHeaderView
                } else {
                    Color.clear.frame(height: 15)
                }
                albumSelectionView
            }
        }
    }

    @ViewBuilder
    var cameraSelectionContainer: some View {
        Group {
            if let cameraSelectionBuilder = cameraSelectionBuilder {
                cameraSelectionBuilder(
                    { viewModel.setPickerMode(.camera) }, // add more
                    { viewModel.onCancelCameraSelection(cameraSelectionService.hasSelected) }, // cancel
                    CameraSelectionView(selectionParamsHolder: selectionParamsHolder)
                )
            } else {
                DefaultCameraSelectionContainer(
                    viewModel: viewModel,
                    showingPicker: $isPresented,
                    selectionParamsHolder: selectionParamsHolder
                )
            }
        }
        .confirmationDialog("", isPresented: $viewModel.showingExitCameraConfirmation, titleVisibility: .hidden) {
            deleteAllButton
        }
    }

    @ViewBuilder
    var cameraContainer: some View {
        ZStack {
            Color.black
                .ignoresSafeArea(.all)
                .onAppear {
                    DispatchQueue.main.async {
                        readyToShowCamera = true
                    }
                }
                .onDisappear {
                    readyToShowCamera = false
                }
            if readyToShowCamera {
                cameraSheet() {
                    // did take picture
                    if !cameraSelectionService.hasSelected {
                        viewModel.setPickerMode(.cameraSelection)
                    }
                    guard let url = viewModel.pickedMediaUrl else { return }
                    cameraSelectionService.onSelect(media: URLMediaModel(url: url))
                    viewModel.pickedMediaUrl = nil
                } didPressCancel: {
                    if let didPressCancel = didPressCancelCamera {
                        didPressCancel()
                    } else {
                        viewModel.setPickerMode(.photos)
                    }
                }
                .confirmationDialog("", isPresented: $viewModel.showingExitCameraConfirmation, titleVisibility: .hidden) {
                    deleteAllButton
                }
            }
        }
        .onAppear {
            orientationHandler(.lock)
        }
        .onDisappear {
            orientationHandler(.unlock)
        }
    }

    var deleteAllButton: some View {
        Button("Delete All") {
            cameraSelectionService.removeAll()
            viewModel.setPickerMode(.photos)
            onChange(selectionService.mapToMedia())
        }
    }

    var defaultHeaderView: some View {
        HStack {
            Button("Cancel") {
                selectionService.removeAll()
                cameraSelectionService.removeAll()
                isPresented = false
            }

            Spacer()

            Picker("", selection:
                    Binding(
                        get: { viewModel.internalPickerMode == .albums ? 1 : 0 },
                        set: { value in
                            viewModel.setPickerMode(value == 0 ? .photos : .albums)
                        }
                    )
            ) {
                Text("Photos")
                    .tag(0)
                Text("Albums")
                    .tag(1)
            }
            .pickerStyle(SegmentedPickerStyle())
            .frame(maxWidth: UIScreen.main.bounds.width / 2)

            Spacer()

            Button("Done") {
                if selectionService.selected.isEmpty, let current = currentFullscreenMedia {
                    onChange([current])
                }
                isPresented = false
            }
        }
        .foregroundColor(theme.main.text)
        .padding(12)
        .background(theme.defaultHeader.background)
    }

    func cameraBinding() -> Binding<Bool> {
        Binding(
            get: { viewModel.internalPickerMode == .camera },
            set: { value in
                if value { viewModel.setPickerMode(.camera) }
            }
        )
    }

    func modeBinding() -> Binding<Int> {
        Binding(
            get: { viewModel.internalPickerMode == .albums ? 1 : 0 },
            set: { value in
                viewModel.setPickerMode(value == 0 ? .photos : .albums)
            }
        )
    }

    @ViewBuilder
    func cameraSheet(didTakePicture: @escaping ()->(), didPressCancel: @escaping ()->()) -> some View {
#if targetEnvironment(simulator)
        CameraStubView {
            didPressCancel()
        }
#elseif os(iOS)
        Group {
            if let cameraViewBuilder = cameraViewBuilder {
                CustomCameraView<CameraViewContent>(viewModel: viewModel, didTakePicture: didTakePicture, didPressCancel: didPressCancel, cameraViewBuilder: cameraViewBuilder)
                    .ignoresSafeArea()
            } else {
                StandardConrolsCameraView(viewModel: viewModel, didTakePicture: didTakePicture, didPressCancel: didPressCancel, selectionParamsHolder: selectionParamsHolder)
                    .ignoresSafeArea()
            }
        }
        .onAppear {
            permissionService.requestCameraPermission()
        }
#endif
    }
}

// MARK: - Customization

public extension MediaPicker {

    func showLiveCameraCell(_ show: Bool = true) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.showingLiveCameraCell = show
        return mediaPicker
    }

    func mediaSelectionType(_ type: MediaSelectionType) -> MediaPicker {
        selectionParamsHolder.mediaType = type
        return self
    }

    func mediaSelectionStyle(_ style: MediaSelectionStyle) -> MediaPicker {
        selectionParamsHolder.selectionStyle = style
        return self
    }

    func mediaSelectionLimit(_ limit: Int) -> MediaPicker {
        selectionParamsHolder.selectionLimit = limit
        return self
    }

    func showFullscreenPreview(_ show: Bool) -> MediaPicker {
        selectionParamsHolder.showFullscreenPreview = show
        return self
    }

    func setSelectionParameters(_ params: SelectionParamsHolder?) -> MediaPicker {
        guard let params = params else {
            return self
        }
        var mediaPicker = self
        mediaPicker.selectionParamsHolder = params
        return mediaPicker
    }

    func applyFilter(_ filterClosure: @escaping FilterClosure) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.filterClosure = filterClosure
        return mediaPicker
    }

    func applyFilter(_ filterClosure: @escaping MassFilterClosure) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.massFilterClosure = filterClosure
        return mediaPicker
    }

    func didPressCancelCamera(_ didPressCancelCamera: @escaping ()->()) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.didPressCancelCamera = didPressCancelCamera
        return mediaPicker
    }

    func orientationHandler(_ orientationHandler: @escaping MediaPickerOrientationHandler) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.orientationHandler = orientationHandler
        return mediaPicker
    }

    func currentFullscreenMedia(_ currentFullscreenMedia: Binding<Media?>) -> MediaPicker {
        var mediaPicker = self
        mediaPicker._currentFullscreenMediaBinding = currentFullscreenMedia
        return mediaPicker
    }

    func albums(_ albums: Binding<[Album]>) -> MediaPicker {
        var mediaPicker = self
        mediaPicker._albums = albums
        return mediaPicker
    }

    func pickerMode(_ mode: Binding<MediaPickerMode>) -> MediaPicker {
        var mediaPicker = self
        mediaPicker.pickerMode = mode
        return mediaPicker
    }
}
