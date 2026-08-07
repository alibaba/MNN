//
//  ContentView.swift
//  ChatFirestoreExample
//
//  Created by Alisa Mylnikova on 12.06.2023.
//

import SwiftUI

struct ContentView: View {

    @AppStorage(hasCurrentSessionKey) var hasCurrentSession = false

    var body: some View {
        Group {
            if hasCurrentSession {
                ConversationsView()
            } else {
                AuthView()
            }
        }
        .onAppear {
            if hasCurrentSession {
                SessionManager.shared.loadUser()
            }
        }
    }
}
