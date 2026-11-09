import Foundation

/// Localizes UIKit and precomputed strings using the app's selected language.
/// SwiftUI text uses the locale supplied by the app's root view.
public enum ChatTextLocalization {
    private final class Storage {
        let lock = NSLock()
        var bundle = Bundle.main
    }

    private static let storage = Storage()

    public static func useLanguage(_ code: String, in resources: Bundle = .main) {
        let bundle = resources.path(forResource: code, ofType: "lproj")
            .flatMap { Bundle(path: $0) } ?? resources
        storage.lock.lock()
        storage.bundle = bundle
        storage.lock.unlock()
    }

    public static func text(_ key: String, defaultValue: String? = nil, comment: String = "") -> String {
        storage.lock.lock()
        let bundle = storage.bundle
        storage.lock.unlock()
        return bundle.localizedString(forKey: key, value: defaultValue, table: nil)
    }
}
