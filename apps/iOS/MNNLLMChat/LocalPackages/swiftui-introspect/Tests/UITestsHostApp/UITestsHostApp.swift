import SwiftUI

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    
    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        guard
            let testCaseRawValue = ProcessInfo.processInfo.environment["testCase"],
            let testCase = TestCase(rawValue: testCaseRawValue)
        else {
            preconditionFailure("entryViewController not set")
        }
        
        window?.rootViewController = {
            switch testCase {
            case .statusBarStyle:
                let navController = NavigationController(rootViewController: HostingController(rootView: RootView()))
                NavigationController.shared = navController
                return navController
            }
        }()
        window?.makeKeyAndVisible()
        return true
    }
}
