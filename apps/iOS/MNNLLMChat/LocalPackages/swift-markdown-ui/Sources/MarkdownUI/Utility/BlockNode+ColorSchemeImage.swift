import SwiftUI

extension Sequence where Element == BlockNode {
  func filterImagesMatching(colorScheme: ColorScheme) -> [BlockNode] {
    self.rewrite { inline in
      switch inline {
      case .image(let source, _):
        guard let url = URL(string: source), url.matchesColorScheme(colorScheme) else {
          return []
        }
        return [inline]
      default:
        return [inline]
      }
    }
  }
}

extension URL {
  fileprivate func matchesColorScheme(_ colorScheme: ColorScheme) -> Bool {
    guard let fragment = self.fragment?.lowercased() else {
      return true
    }

    switch colorScheme {
    case .light:
      return fragment != "gh-dark-mode-only"
    case .dark:
      return fragment != "gh-light-mode-only"
    @unknown default:
      return true
    }
  }
}
