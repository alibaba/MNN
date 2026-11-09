//
//  AppDelegate.swift
//  Example-iOS
//
//  Created by Alexandra Afonasova on 20.10.2022.
//

import Foundation
import UIKit

final class AppDelegate: NSObject, UIApplicationDelegate, ObservableObject {

    private var orientationLock = UIInterfaceOrientationMask.all

    func lockOrientationToPortrait() {
        orientationLock = .portrait
        if #available(iOS 16, *) {
            if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene {
                scene.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait))
            }
            UIViewController.attemptRotationToDeviceOrientation()
        } else {
            UIDevice.current.setValue(UIDeviceOrientation.portrait.rawValue, forKey: "orientation")
        }
    }

    func unlockOrientation() {
        orientationLock = .all
        UIViewController.attemptRotationToDeviceOrientation()
    }

    func application(_ application: UIApplication, supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        return orientationLock
    }

}
