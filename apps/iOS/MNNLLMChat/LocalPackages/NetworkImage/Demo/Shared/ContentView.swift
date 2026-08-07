import SwiftUI

struct ContentView: View {
  var body: some View {
    NavigationView {
      List {
        NavigationLink {
          ImagesView()
            .navigationTitle("Displaying Images")
            .inlineNavigationBarTitleDisplayMode()
        } label: {
          Label("Displaying Images", systemImage: "photo")
        }
        NavigationLink {
          StyledImagesView()
            .navigationTitle("Styling Images")
            .inlineNavigationBarTitleDisplayMode()
        } label: {
          Label("Styling Images", systemImage: "photo.artframe")
        }
        NavigationLink {
          PlaceholdersView()
            .navigationTitle("Placeholders and fallbacks")
            .inlineNavigationBarTitleDisplayMode()
        } label: {
          Label("Placeholders and fallbacks", systemImage: "square.dashed")
        }
        NavigationLink {
          RandomImageView()
            .navigationTitle("Random Image")
            .inlineNavigationBarTitleDisplayMode()
        } label: {
          Label("Random Image", systemImage: "questionmark.square")
        }
        NavigationLink {
          ImageListView()
            .navigationTitle("Image List")
            .inlineNavigationBarTitleDisplayMode()
        } label: {
          Label("Image List", systemImage: "scroll")
        }
      }
      .navigationTitle("NetworkImage")
    }
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
