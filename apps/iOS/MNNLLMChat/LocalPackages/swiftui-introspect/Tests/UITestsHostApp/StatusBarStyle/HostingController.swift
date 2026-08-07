//
//  HostingController.swift
//  Blago
//
//  Created by Dmytro Chumakov on 04.05.2023.
//

import SwiftUI

final class HostingController<ContentView>: UIHostingController<ContentView> where ContentView: View {

    var statusBarStyle: UIStatusBarStyle = .darkContent
    var isInteractivePopGestureEnabled = true

    override var preferredStatusBarStyle: UIStatusBarStyle {
        statusBarStyle
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        navigationController?.interactivePopGestureRecognizer?.isEnabled = isInteractivePopGestureEnabled
    }

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()

        guard #available(iOS 16, *) else {
            navigationController?.setNavigationBarHidden(true, animated: false)
            return
        }
    }
}
