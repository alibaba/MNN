//
//  ConversationsView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 03.07.2023.
//

import SwiftUI

struct ConversationsView: View {

    @ObservedObject var dataStorage = DataStorageManager.shared
    @StateObject var viewModel = ConversationsViewModel()
    @StateObject var networkMonitor = NetworkMonitor()

    @State var showUsersList = false
    @State var navPath = NavigationPath()

    var body: some View {
        ZStack {
            content

            if viewModel.showActivityIndicator {
                ActivityIndicator()
            }
        }
        .task {
            viewModel.subscribeToUpdates()
        }
    }

    var content: some View {
        NavigationStack(path: $navPath) {
            if !networkMonitor.isConnected {
                waitingForNetwork
            }
            
            SearchField(text: $viewModel.searchText)
                .padding(.horizontal, 12)

            List(viewModel.filteredConversations) { conversation in
                HStack {
                    if let url = conversation.pictureURL {
                        AvatarView(url: url, size: 56)
                    } else {
                        HStack(spacing: -30) {
                            ForEach(conversation.notMeUsers) { user in
                                AvatarView(url: user.avatarURL, size: 56)
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text(conversation.displayTitle)
                            .font(17, .black, .medium)

                        if let latest = conversation.latestMessage {
                            HStack(spacing: 0) {
                                if conversation.isGroup {
                                    Text("\(latest.senderName): ")
                                        .font(15, .exampleTertiaryText)
                                } else if latest.isMyMessage {
                                    Text("You: ")
                                        .font(15, .exampleTertiaryText)
                                }

                                HStack(spacing: 4) {
                                    if let subtext = latest.subtext {
                                        Text(subtext)
                                            .font(15, .exampleBlue)
                                    }
                                    if let text = latest.text {
                                        Text(text)
                                            .lineLimit(1)
                                            .font(15, .exampleSecondaryText)
                                    }
                                    if let date = latest.createdAt?.timeAgoFormat() {
                                        Text("·")
                                            .font(13, .exampleTertiaryText)
                                        Text(date)
                                            .font(13, .exampleTertiaryText)
                                    }
                                }
                            }
                        }
                    }

                    Spacer()

                    if let unreadCounter = conversation.usersUnreadCountInfo[SessionManager.currentUserId], unreadCounter != 0 {
                        Text("\(unreadCounter)")
                            .font(15, .white, .semibold)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background {
                                Color.exampleBlue.cornerRadius(.infinity)
                            }
                    }
                }
                .background(
                    NavigationLink("", value: conversation)
                        .opacity(0)
                )
                .listRowSeparator(.hidden)
            }
            .refreshable {
                await viewModel.getData()
            }
            .listStyle(.plain)
            .navigationTitle("Chats")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: Conversation.self) { conversation in
                ConversationView(viewModel: ConversationViewModel(conversation: conversation))
            }
            .navigationDestination(for: User.self) { user in
                ConversationView(viewModel: ConversationViewModel(user: user))
            }
            .toolbar {
                ToolbarItem {
                    Button {
                        showUsersList = true
                    } label: {
                        Image(.newChat)
                    }
                }
            }
            .toolbar {
                ToolbarItem(placement: .navigation) {
                    Button("Log out") {
                        SessionManager.shared.logout()
                    }
                    .font(17, .black)
                    .padding(.leading, 8)
                }
            }
        }
        .sheet(isPresented: $showUsersList) {
            UsersView(viewModel: UsersViewModel(), isPresented: $showUsersList, navPath: $navPath)
        }
    }

    var waitingForNetwork: some View {
        VStack {
            Rectangle()
                .foregroundColor(.black.opacity(0.12))
                .frame(height: 1)
            HStack {
                Spacer()
                Image("waiting", bundle: .current)
                Text("Waiting for network")
                Spacer()
            }
            .padding(.top, 6)
            Rectangle()
                .foregroundColor(.black.opacity(0.12))
                .frame(height: 1)
        }
        .padding(.top, 8)
    }
}
