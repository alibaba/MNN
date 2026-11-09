//
//  Created by Alex.M on 06.06.2022.
//

import Foundation
import SwiftUI

struct PermissionsActionView: View {

    let action: Action
    
    @State private var showSheet = false
    
    var body: some View {
        ZStack {
            if showSheet {
                LimitedLibraryPickerProxyView(isPresented: $showSheet) {
                    NotificationCenter.default.post(
                        name: photoLibraryChangeLimitedPhotosNotification,
                        object: nil)
                }
                .frame(width: 1, height: 1)
            }
            
            switch action {
            case .library(let assetsLibraryAction):
                buildLibraryAction(assetsLibraryAction)
            case .camera(let cameraAction):
                buildCameraAction(cameraAction)
            }
        }
    }
}

private extension PermissionsActionView {
    
    @ViewBuilder
    func buildLibraryAction(_ action: PermissionsService.PhotoLibraryAction) -> some View {
        switch action {
        case .selectMore:
            PermissionsErrorView(text: "Setup Photos access to see more photos here") {
                showSheet = true
            }
        case .authorize:
            goToSettingsButton(text: "Allow Photos access in settings to see photos here")
        case .unavailable:
            PermissionsErrorView(text: "Sorry, Photos are not available.", action: nil)
        case .unknown:
            fatalError("Unknown permission status.")
        }
    }
    
    @ViewBuilder
    func buildCameraAction(_ action: PermissionsService.CameraAction) -> some View {
        switch action {
        case .authorize:
            goToSettingsButton(text: "Allow Camera access in settings to see live preview")
        case .unavailable:
            PermissionsErrorView(text: "Sorry, Camera is not available.", action: nil)
        case .unknown:
            fatalError("Unknown permission status.")
        }
    }
    
    func goToSettingsButton(text: String) -> some View {
        PermissionsErrorView(
            text: text,
            action: {
                guard let url = URL(string: UIApplication.openSettingsURLString)
                else { return }
                if UIApplication.shared.canOpenURL(url) {
                    UIApplication.shared.open(url)
                }
            }
        )
    }
}

extension PermissionsActionView {
    enum Action {
        case library(PermissionsService.PhotoLibraryAction)
        case camera(PermissionsService.CameraAction)
    }
}
