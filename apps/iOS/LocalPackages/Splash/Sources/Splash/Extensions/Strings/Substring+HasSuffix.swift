import Foundation

#if os(Linux)

internal extension Substring {
    func hasSuffix(_ suffix: String) -> Bool {
        guard count >= suffix.count else {
            return false
        }

        let startIndex = index(endIndex, offsetBy: -suffix.count)
        return self[startIndex...] == suffix
    }
}

#endif
