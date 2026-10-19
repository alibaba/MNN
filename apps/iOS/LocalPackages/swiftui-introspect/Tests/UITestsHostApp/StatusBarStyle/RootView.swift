//
//  RootView.swift
//  Showcase
//
//  Created by Orackle on 14.07.2023.
//

import SwiftUI
import SwiftUIIntrospect

struct RootView: View {

    var body: some View {
        VStack {
            Button("Navigate To Detail", action: navigateToDetail)
        }
    }

    @MainActor // for below Swift 6.0
    private func navigateToDetail() {
        let controller = HostingController(rootView: DetailView())
        controller.statusBarStyle = .lightContent
        NavigationController.shared?.pushViewController(controller, animated: true)
    }
}

struct DetailView: View {
    @Environment(\.presentationMode) var dismiss

    var body: some View {
        ZStack {
            Color.red.edgesIgnoringSafeArea(.all)

            VStack {
                Button("Navigate To Detail", action: navigateToDetail)
                Button("Navigate Back", action: goBack)
            }
        }
        .introspect(.viewController, on: .iOS(.v13, .v14, .v15, .v16)) { viewController in
            /// some customizations there
        }
    }

    private func goBack() {
        dismiss.wrappedValue.dismiss()
    }

    @MainActor // for below Swift 6.0
    private func navigateToDetail() {
        let controller = HostingController(rootView: DetailView())
        controller.statusBarStyle = .lightContent
        NavigationController.shared?.pushViewController(controller, animated: true)
    }
}
