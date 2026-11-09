//
//  NewGroupConversationView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 10.07.2023.
//

import SwiftUI

struct GroupSelectUsersView: View {

    @Environment(\.presentationMode) var presentationMode

    @StateObject var viewModel: UsersViewModel
    @Binding var isPresented: Bool
    @Binding var navPath: NavigationPath

    var body: some View {
        VStack {
            SearchField(text: $viewModel.searchableText)
                .padding(.horizontal, 16)

            ZStack(alignment: .bottom) {
                List(viewModel.filteredUsers) { user in
                    HStack {
                        AvatarView(url: user.avatarURL, size: 48)
                        Text(user.name)
                            .font(17, .black, .medium)
                        
                        Spacer()
                        
                        (viewModel.selectedUsers.contains(user) ? Image(.checkSelected) : Image(.checkUnselected))
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        viewModel.selectedUsers.contains(user) ?
                        viewModel.selectedUsers.removeAll(where: { $0.id == user.id }) :
                        viewModel.selectedUsers.append(user)
                    }
                    .listRowSeparator(.hidden)
                }
                .padding(.bottom, 60)

                NavigationLink("Next") {
                    GroupCreateView(viewModel: viewModel, isPresented: $isPresented, navPath: $navPath)
                }
                .buttonStyle(BlueButton())
                .disabled(viewModel.selectedUsers.count < 1)
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
            }
        }
        .listStyle(.plain)
        .navigationTitle("Create Group")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden()
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    presentationMode.wrappedValue.dismiss()
                } label: {
                    Image(.navigateBack)
                }
            }
            // HStack in ToolbarItem doesn't work, so add a second button
            ToolbarItem(placement: .navigation) {
                Button {
                    presentationMode.wrappedValue.dismiss()
                } label: {
                    Text("Back")
                        .font(17, .black)
                }
            }
        }
    }
}
