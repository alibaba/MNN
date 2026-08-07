//
//  AuthView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 12.06.2023.
//

import SwiftUI
import ExyteMediaPicker

struct AuthView: View {

    @StateObject var viewModel = AuthViewModel()

    @State var name: String = ""

    @State var showPicker = false
    @State var avatar: Media?
    @State var avatarURL: URL?

    var body: some View {
        ZStack {
            content

            if viewModel.showActivityIndicator {
                ActivityIndicator()
            }
        }
    }

    var content: some View {
        ZStack {
            VStack(spacing: 0) {
                Text("Welcome!\nCreate a test account")
                    .font(20, .black, .medium)
                    .multilineTextAlignment(.center)
                    .fixedSize()
                
                pickAvatarView
                    .padding(.top, 80)
                
                CustomTextField(placeholder: "Nickname", text: $name)
                    .padding(.top, 44)
                
                createButton
                    .padding(.top, 16)
                
                Spacer()
                
                Image(.logo)
            }
            .padding(12)
            .padding(.top, 70)
        }
        .contentShape(Rectangle())
        .onTapGesture {
            hideKeyboard()
        }
        .fullScreenCover(isPresented: $showPicker) {
            MediaPicker(isPresented: $showPicker) { media in
                avatar = media.first
                Task {
                    await avatarURL = avatar?.getURL()
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

    var pickAvatarView: some View {
        ZStack(alignment: .bottomTrailing) {
            AsyncImage(url: avatarURL) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                ZStack {
                    Color.exampleLightGray
                    Image(.avatarPlaceholder)
                }
            }
            .frame(width: 146, height: 146)
            .clipShape(Circle())

            ZStack {
                Circle()
                    .strokeBorder(Color.white, lineWidth: 3)
                    .background(Circle().foregroundColor(Color.exampleDarkGray))
                    .frame(width: 48, height: 48)
                Image(.photoIcon)
            }
        }
        .onTapGesture {
            showPicker = true
        }
    }

    var createButton: some View {
        Button {
            viewModel.auth(nickname: name, avatar: avatar)
        } label: {
            Text("Create")
                .font(17, .white, .medium)
                .frame(height: 48)
                .frame(maxWidth: .infinity)
                .background {
                    RoundedRectangle(cornerRadius: 14)
                        .foregroundColor(name.isEmpty ? .exampleMidGray : .exampleBlue)
                }
        }
        .disabled(name.isEmpty)
    }
}
