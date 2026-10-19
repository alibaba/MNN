import SwiftUI

struct ThematicBreakView: View {
  @Environment(\.theme.thematicBreak) private var thematicBreak

  var body: some View {
    self.thematicBreak.makeBody(configuration: ())
  }
}
