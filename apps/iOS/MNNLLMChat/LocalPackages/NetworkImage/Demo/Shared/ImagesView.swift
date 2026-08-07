import NetworkImage
import SwiftUI

struct ImagesView: View {
  var body: some View {
    Form {
      NetworkImage(url: URL(string: "https://picsum.photos/id/1025/300/200"))
        .frame(width: 200, height: 200)
        .clipShape(RoundedRectangle(cornerRadius: 8))
      NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200"))
        .frame(width: 200, height: 200)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
  }
}

struct ImagesView_Previews: PreviewProvider {
  static var previews: some View {
    ImagesView()
  }
}
