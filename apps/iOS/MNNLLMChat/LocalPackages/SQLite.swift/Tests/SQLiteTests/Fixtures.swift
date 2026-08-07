import Foundation

func fixture(_ name: String, withExtension: String?) -> String {
    #if SWIFT_PACKAGE
    let testBundle = Bundle.module
    #else
    let testBundle = Bundle(for: SQLiteTestCase.self)
    #endif

    for resource in [name, "Resources/\(name)"] {
        if let url = testBundle.url(
            forResource: resource,
            withExtension: withExtension) {
            return url.path
        }
    }
    fatalError("Cannot find \(name).\(withExtension ?? "")")
}

func temporaryFile() -> String {
    URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString).path
}
