//
//  UsersView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 12.06.2023.
//

import SwiftUI

struct UsersView: View {

    @Environment(\.presentationMode) var presentationMode
    @ObservedObject var dataStorage = DataStorageManager.shared

    @StateObject var viewModel: UsersViewModel
    @Binding var isPresented: Bool
    @Binding var navPath: NavigationPath

    @State var showActivityIndicator = false

    var body: some View {
        ZStack {
            content

            if showActivityIndicator {
                ActivityIndicator()
            }
        }
    }

    var content: some View {
        NavigationStack {
            SearchField(text: $viewModel.searchableText)
                .padding(.horizontal, 16)

            List {
                NavigationLink {
                    GroupSelectUsersView(viewModel: viewModel, isPresented: $isPresented, navPath: $navPath)
                } label: {
                    HStack {
                        ZStack {
                            Circle()
                                .foregroundColor(.exampleBlue)
                            Image(.groupChat)
                        }
                        .frame(width: 48, height: 48)
                        Text("Create Group")
                            .font(17, .exampleBlue, .medium)
                    }
                }
                .listRowSeparator(.hidden)
                .listRowInsets(EdgeInsets())
                .padding(.top, 8)
                .padding(.horizontal, 16)

                Rectangle()
                    .foregroundColor(.exampleFieldBorder)
                    .frame(height: 1)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets())

                ForEach(viewModel.filteredUsers) { user in
                    HStack {
                        AvatarView(url: user.avatarURL, size: 48)
                        Text(user.name)
                            .font(17, .black, .medium)
                    }
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets())
                    .padding([.horizontal, .bottom], 16)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        Task {
                            if let conversation = await viewModel.conversationForUsers([user]) {
                                navPath.append(conversation)
                            } else {
                                navPath.append(user)
                            }
                            isPresented = false
                        }
                    }
                }
            }
            .listStyle(.plain)
            .navigationTitle("New Message")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                    .font(17, .black)
                }
            }
        }
    }
}
