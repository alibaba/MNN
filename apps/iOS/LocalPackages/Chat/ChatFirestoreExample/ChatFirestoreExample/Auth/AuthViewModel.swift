//
//  AuthViewModel.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 12.06.2023.
//

import Foundation
import FirebaseFirestore
import FirebaseFirestoreSwift
import ExyteMediaPicker

class AuthViewModel: ObservableObject {

    @Published var showActivityIndicator = false

    func auth(nickname: String, avatar: Media?) {
        showActivityIndicator = true
        Firestore.firestore()
            .collection(Collection.users)
            .whereField("deviceId", isEqualTo: SessionManager.shared.deviceId)
            .whereField("nickname", isEqualTo: nickname)
            .getDocuments { [weak self] (snapshot, _) in
                let data = snapshot?.documents.first?.data()
                if let id = snapshot?.documents.first?.documentID {
                    var url: URL? = nil
                    if let string = data?["avatarURL"] as? String {
                        url = URL(string: string)
                    }
                    let user = User(id: id, name: nickname, avatarURL: url, isCurrentUser: true)
                    SessionManager.shared.storeUser(user)
                    self?.showActivityIndicator = false
                } else {
                    self?.createNewUser(nickname: nickname, avatar: avatar)
                }
            }
    }

    func createNewUser(nickname: String, avatar: Media?) {
        Task {
            let avatarURL = await UploadingManager.uploadImageMedia(avatar)
            var ref: DocumentReference? = nil
            ref = Firestore.firestore()
                .collection(Collection.users).addDocument(data: [
                    "deviceId": SessionManager.shared.deviceId,
                    "nickname": nickname,
                    "avatarURL": avatarURL?.absoluteString as Any
                ]) { [weak self] err in
                    if let err = err {
                        print("Error adding document: \(err)")
                    } else if let id = ref?.documentID {
                        let user = User(id: id, name: nickname, avatarURL: avatarURL, isCurrentUser: true)
                        SessionManager.shared.storeUser(user)
                    }
                    self?.showActivityIndicator = false
                }
        }
    }
}
