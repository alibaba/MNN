//
//  GroupCreateView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 10.07.2023.
//

import SwiftUI
import ExyteMediaPicker

struct GroupCreateView: View {

    @Environment(\.presentationMode) var presentationMode

    @StateObject var viewModel: UsersViewModel
    @Binding var isPresented: Bool
    @Binding var navPath: NavigationPath

    // private

    @State private var showPicker = false
    @State private var avatarURL: URL?

    @State private var showActivityIndicator = false

    var body: some View {
        ZStack {
            content

            if showActivityIndicator {
                ActivityIndicator()
            }
        }
    }

    var content: some View {
        VStack {
            pickImageView
                .padding(.top, 32)

            CustomTextField(placeholder: "Group name", text: $viewModel.title)
                .padding(.top, 32)

            Spacer()

            Button("Create") {
                Task {
                    showActivityIndicator = true
                    if let conversation = await viewModel.createConversation(viewModel.selectedUsers) {
                        showActivityIndicator = false
                        viewModel.selectedUsers = []
                        isPresented = false
                        navPath.append(conversation)
                    }
                }
            }
            .buttonStyle(BlueButton())
            .disabled(viewModel.title.isEmpty)
            .padding(.bottom, 10)
        }
        .padding(.horizontal, 12)
        .navigationTitle("Create Group")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                }
                .font(17, .black)
            }
        }
        .fullScreenCover(isPresented: $showPicker) {
            MediaPicker(isPresented: $showPicker) { media in
                viewModel.picture = media.first
                Task {
                    await avatarURL = viewModel.picture?.getURL()
                }
            }
            .mediaSelectionLimit(1)
            .mediaSelectionType(.photo)
            .showLiveCameraCell()
            .orientationHandler {
                switch $0 {
                case .lock: AppDelegate.lockOrientationToPortrait()
                case .unlock: AppDelegate.unlockOrientation()
                }
            }
        }
    }

    var pickImageView: some View {
        AsyncImage(url: avatarURL) { image in
            image
                .resizable()
                .aspectRatio(contentMode: .fill)
        } placeholder: {
            ZStack {
                Color.exampleLightGray
                Image(.imagePlaceholder)
            }
        }
        .frame(width: 120, height: 120)
        .clipShape(Circle())
        .contentShape(Circle())
        .onTapGesture {
            showPicker = true
        }
    }
}
