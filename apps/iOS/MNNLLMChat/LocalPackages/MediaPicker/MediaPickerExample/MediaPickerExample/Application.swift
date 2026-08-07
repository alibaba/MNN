//
//  Example_iOSApp.swift
//  Example-iOS
//
//  Created by Alex.M on 26.05.2022.
//

import SwiftUI

@main
struct Application: App {

    @UIApplicationDelegateAdaptor var appDelegate: AppDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appDelegate)
        }
    }
}
