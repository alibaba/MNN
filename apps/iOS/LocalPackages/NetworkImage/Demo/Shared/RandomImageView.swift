import NetworkImage
import SwiftUI

struct RandomImageView: View {
  @State var url = URL.randomImageURL(size: .init(width: 300, height: 200))

  var body: some View {
    Form {
      Section {
        Text("This example shows how to transition between different images by changing the URL.")
      }
      NetworkImage(url: self.url, transaction: .init(animation: .default)) { image in
        image.resizable().scaledToFill()
      }
      .frame(width: 200, height: 200)
      .clipShape(RoundedRectangle(cornerRadius: 8))

      Button("Random Image") {
        self.url = URL.randomImageURL(size: .init(width: 300, height: 200))
      }
    }
  }
}

struct RandomImageView_Previews: PreviewProvider {
  static var previews: some View {
    RandomImageView()
  }
}
