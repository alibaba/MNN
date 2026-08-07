# NetworkImage
[![CI](https://github.com/gonzalezreal/NetworkImage/workflows/CI/badge.svg)](https://github.com/gonzalezreal/NetworkImage/actions?query=workflow%3ACI)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fgonzalezreal%2FNetworkImage%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/gonzalezreal/NetworkImage)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fgonzalezreal%2FNetworkImage%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/gonzalezreal/NetworkImage)

NetworkImage is a Swift package that provides image downloading, caching, and displaying for your SwiftUI apps. It leverages the foundation `URLCache` and `NSCache`, providing persistent and in-memory caches.

Explore the [companion demo project](Demo) to discover its capabilities.

## Minimum requirements

You can use NetworkImage on the following platforms:

* macOS 11.0+
* iOS 14.0+
* tvOS 14.0+
* watchOS 7.0+

## Usage

A network image downloads and displays an image from a given URL; the download is asynchronous,
and the result is cached both in disk and memory.

The simplest way of creating a `NetworkImage` view is to pass the image URL to the `init(url:scale:)` initializer.

```swift
NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200"))
  .frame(width: 300, height: 200)
```

To manipulate the loaded image, use the `content` parameter.

```swift
NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { image in
  image
    .resizable()
    .scaledToFill()
    .blur(radius: 4)
}
.frame(width: 150, height: 150)
.clipped()
```

The view displays a standard placeholder that fills the available space until the image loads. You
can specify a custom placeholder by using the `placeholder` parameter.

```swift
NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { image in
  image
    .resizable()
    .scaledToFill()
} placeholder: {
  ZStack {
    Color.secondary.opacity(0.25)
    Image(systemName: "photo.fill")
      .imageScale(.large)
      .blendMode(.overlay)
  }
}
.frame(width: 150, height: 150)
.clipped()
```

To have more control over the image loading process, use the `init(url:scale:transaction:content)` initializer, which takes a `content` closure that receives a `NetworkImageState` to indicate the state of the loading operation.

```swift
NetworkImage(url: URL(string: "https://picsum.photos/id/237/300/200")) { state in
  switch state {
  case .empty:
    ProgressView()
  case .success(let image, let idealSize):
    image
      .resizable()
      .scaledToFill()
  case .failure:
    Image(systemName: "photo.fill")
      .imageScale(.large)
      .blendMode(.overlay)
  }
}
.frame(width: 150, height: 150)
.background(Color.secondary.opacity(0.25))
.clipped()
```

## Installation
### Adding NetworkImage to a Swift package

To use NetworkImage in a Swift Package Manager project, add the following line to the dependencies in your `Package.swift` file:

```swift
.package(url: "https://github.com/gonzalezreal/NetworkImage", from: "6.0.0")
```

Include `"NetworkImage"` as a dependency for your executable target:

```swift
.target(name: "<target>", dependencies: [
  .product(name: "NetworkImage", package: "NetworkImage")
]),
```

Finally, add `import NetworkImage` to your source code.

### Adding NetworkImage to an Xcode project

1. From the **File** menu, select **Add Packages…**
1. Enter `https://github.com/gonzalezreal/NetworkImage` into the
   *Search or Enter Package URL* search field
1. Link **NetworkImage** to your application target
