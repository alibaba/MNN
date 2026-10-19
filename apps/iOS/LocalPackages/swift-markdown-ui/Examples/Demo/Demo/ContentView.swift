import SwiftUI

struct ContentView: View {
  var body: some View {
    NavigationView {
      Form {
        Section("Formatting") {
          NavigationLink {
            HeadingsView()
              .navigationTitle("Headings")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Headings", systemImage: "textformat.size")
          }
          NavigationLink {
            ListsView()
              .navigationTitle("Lists")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Lists", systemImage: "list.bullet")
          }
          NavigationLink {
            TextStylesView()
              .navigationTitle("Text Styles")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Text Styles", systemImage: "textformat.abc")
          }
          NavigationLink {
            QuotesView()
              .navigationTitle("Quotes")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Quotes", systemImage: "text.quote")
          }
          NavigationLink {
            CodeView()
              .navigationTitle("Code")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Code", systemImage: "curlybraces")
          }
          NavigationLink {
            ImagesView()
              .navigationTitle("Images")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Images", systemImage: "photo")
          }
          NavigationLink {
            TablesView()
              .navigationTitle("Tables")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Tables", systemImage: "tablecells")
          }
        }
        Section("Extensibility") {
          NavigationLink {
            CodeSyntaxHighlightView()
              .navigationTitle("Syntax Highlighting")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Syntax Highlighting", systemImage: "circle.grid.cross.left.filled")
          }
          NavigationLink {
            ImageProvidersView()
              .navigationTitle("Image Providers")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Image Providers", systemImage: "powerplug")
          }
        }
        Section("Other") {
          NavigationLink {
            DingusView()
              .navigationTitle("Dingus")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Dingus", systemImage: "character.cursor.ibeam")
          }
          NavigationLink {
            RepositoryReadmeView()
              .navigationTitle("Repository README")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Repository README", systemImage: "doc.text")
          }
          NavigationLink {
            LazyLoadingView()
              .navigationTitle("Lazy Loading")
              .navigationBarTitleDisplayMode(.inline)
          } label: {
            Label("Lazy Loading", systemImage: "scroll")
          }
        }
      }
      .navigationTitle("MarkdownUI")
    }
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
