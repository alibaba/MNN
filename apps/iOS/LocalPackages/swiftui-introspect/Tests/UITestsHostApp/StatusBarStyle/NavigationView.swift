
//
//  NavigationController.swift
//  Blago
//
//  Created by Dmytro Chumakov on 11.05.2023.
//

import UIKit

// MARK: - NavigationController

final class NavigationController: UINavigationController {

    static var shared: UINavigationController?

    // MARK: Lifecycle

    override init(rootViewController: UIViewController) {
        super.init(rootViewController: rootViewController)
        setNavigationBarHidden(true, animated: false)
    }

    required init?(coder _: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var preferredStatusBarStyle: UIStatusBarStyle {
        topViewController?.preferredStatusBarStyle ?? .default
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        interactivePopGestureRecognizer?.delegate = self
    }
}

// MARK: UIGestureRecognizerDelegate

extension NavigationController: UIGestureRecognizerDelegate {
    func gestureRecognizerShouldBegin(_: UIGestureRecognizer) -> Bool {
        viewControllers.count > 1
    }
}
