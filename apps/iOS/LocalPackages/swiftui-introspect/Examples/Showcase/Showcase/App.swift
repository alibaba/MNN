import SwiftUI

#if os(iOS) || os(tvOS)
@main
final class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = UIHostingController(rootView: AppView())
        window?.makeKeyAndVisible()
        return true
    }
}
#elseif os(macOS) || os(visionOS)
@main
struct App: SwiftUI.App {
    var body: some Scene {
        WindowGroup {
            AppView()
        }
    }
}
#endif

#if swift(>=5.9)
#Preview {
    AppView()
}
#endif
