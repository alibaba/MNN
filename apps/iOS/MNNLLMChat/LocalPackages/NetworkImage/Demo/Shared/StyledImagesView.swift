import NetworkImage
import SwiftUI

struct StyledImagesView: View {
  var body: some View {
    Form {
      NetworkImage(url: URL(string: "https://picsum.photos/id/1025/300/200")) { image in
        image
          .resizable()
          .scaledToFill()
          .grayscale(1)
      }
      .frame(width: 200, height: 200)
      .clipShape(RoundedRectangle(cornerRadius: 8))
      NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { image in
        image
          .resizable()
          .scaledToFill()
          .blur(radius: 4)
      }
      .frame(width: 200, height: 200)
      .clipShape(RoundedRectangle(cornerRadius: 8))
    }
  }
}

struct StyledImagesView_Previews: PreviewProvider {
  static var previews: some View {
    StyledImagesView()
  }
}
