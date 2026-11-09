//
//  DataStorageManager.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 13.07.2023.
//

import Foundation
import FirebaseFirestore

class DataStorageManager: ObservableObject {

    static var shared = DataStorageManager()

    @Published var users: [User] = [] // not including current user
    @Published var allUsers: [User] = []

    @Published var conversations: [Conversation] = []

    func getUsers() async {
        let snapshot = try? await Firestore.firestore()
            .collection(Collection.users)
            .getDocuments()
        storeUsers(snapshot)
    }

    func getConversations() async {
        let snapshot = try? await Firestore.firestore()
            .collection(Collection.conversations)
            .whereField("users", arrayContains: SessionManager.currentUserId)
            .getDocuments()
        storeConversations(snapshot)
    }

    func subscribeToUpdates() {
        Firestore.firestore()
            .collection(Collection.users)
            .addSnapshotListener { [weak self] (snapshot, _) in
                guard let self else { return }
                self.storeUsers(snapshot)
                Task {
                    await self.getConversations() // update in case some new user didn't make it in time for conversations subscription
                }
            }

        Firestore.firestore()
            .collection(Collection.conversations)
            .whereField("users", arrayContains: SessionManager.currentUserId)
            .addSnapshotListener() { [weak self] (snapshot, _) in
                self?.storeConversations(snapshot)
            }
    }

    private func storeUsers(_ snapshot: QuerySnapshot?) {
        guard let currentUser = SessionManager.currentUser else { return }
        DispatchQueue.main.async { [weak self] in
            let users: [User] = snapshot?.documents
                .compactMap { document in
                    let dict = document.data()
                    if document.documentID == currentUser.id {
                        return nil // skip current user
                    }
                    if let name = dict["nickname"] as? String {
                        let avatarURL = dict["avatarURL"] as? String
                        return User(id: document.documentID, name: name, avatarURL: URL(string: avatarURL ?? ""), isCurrentUser: false)
                    }
                    return nil
                } ?? []

            self?.users = users
            self?.allUsers = users + [currentUser]
        }
    }

    private func storeConversations(_ snapshot: QuerySnapshot?) {
        DispatchQueue.main.async { [weak self] in
            self?.conversations = snapshot?.documents
                .compactMap { [weak self] document in
                    do {
                        let firestoreConversation = try document.data(as: FirestoreConversation.self)
                        return self?.makeConversation(document.documentID, firestoreConversation)
                    } catch {
                        print(error)
                    }

                    return nil
                }.sorted {
                    if let date1 = $0.latestMessage?.createdAt, let date2 = $1.latestMessage?.createdAt {
                        return date1 > date2
                    }
                    return $0.displayTitle < $1.displayTitle
                }
            ?? []
        }
    }

    private func makeConversation(_ id: String, _ firestoreConversation: FirestoreConversation) -> Conversation {
        var message: LatestMessageInChat? = nil
        if let flm = firestoreConversation.latestMessage,
           let user = allUsers.first(where: { $0.id == flm.userId }) {
            var subtext: String?
            if !flm.attachments.isEmpty, let first = flm.attachments.first {
                subtext = first.type.title
            } else if flm.recording != nil {
                subtext = "Voice recording"
            }
            message = LatestMessageInChat(
                senderName: user.name,
                createdAt: flm.createdAt,
                text: flm.text.isEmpty ? nil : flm.text,
                subtext: subtext
            )
        }
        let users = firestoreConversation.users.compactMap { id in
            allUsers.first(where: { $0.id == id })
        }
        let conversation = Conversation(
            id: id,
            users: users,
            usersUnreadCountInfo: firestoreConversation.usersUnreadCountInfo,
            isGroup: firestoreConversation.isGroup,
            pictureURL: firestoreConversation.pictureURL?.toURL(),
            title: firestoreConversation.title,
            latestMessage: message
        )
        return conversation
    }
}
