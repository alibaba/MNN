#if !os(watchOS)
import SwiftUI

/// An abstract representation of the `Map` type in SwiftUI.
///
/// ### iOS
///
/// ```swift
/// struct ContentView: View {
///     @State var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 51.507222, longitude: -0.1275), span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5))
///
///     var body: some View {
///         Map(coordinateRegion: $region)
///             .introspect(.map, on: .iOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // MKMapView
///             }
///     }
/// }
/// ```
///
/// ### tvOS
///
/// ```swift
/// struct ContentView: View {
///     @State var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 51.507222, longitude: -0.1275), span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5))
///
///     var body: some View {
///         Map(coordinateRegion: $region)
///             .introspect(.map, on: .tvOS(.v14, .v15, .v16, .v17, .v18)) {
///                 print(type(of: $0)) // MKMapView
///             }
///     }
/// }
/// ```
///
/// ### macOS
///
/// ```swift
/// struct ContentView: View {
///     @State var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 51.507222, longitude: -0.1275), span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5))
///
///     var body: some View {
///         Map(coordinateRegion: $region)
///             .introspect(.map, on: .macOS(.v11, .v12, .v13, .v14, .v15)) {
///                 print(type(of: $0)) // MKMapView
///             }
///     }
/// }
/// ```
///
/// ### visionOS
///
/// ```swift
/// struct ContentView: View {
///     @State var region = MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 51.507222, longitude: -0.1275), span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5))
///
///     var body: some View {
///         Map(coordinateRegion: $region)
///             .introspect(.map, on: .visionOS(.v1, .v2)) {
///                 print(type(of: $0)) // MKMapView
///             }
///     }
/// }
/// ```
public struct MapType: IntrospectableViewType {}

#if canImport(MapKit)
import MapKit

extension IntrospectableViewType where Self == MapType {
    public static var map: Self { .init() }
}

extension iOSViewVersion<MapType, MKMapView> {
    @available(*, unavailable, message: "Map isn't available on iOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension tvOSViewVersion<MapType, MKMapView> {
    @available(*, unavailable, message: "Map isn't available on tvOS 13")
    public static let v13 = Self.unavailable()
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
    public static let v16 = Self(for: .v16)
    public static let v17 = Self(for: .v17)
    public static let v18 = Self(for: .v18)
}

extension macOSViewVersion<MapType, MKMapView> {
    @available(*, unavailable, message: "Map isn't available on macOS 10.15")
    public static let v10_15 = Self.unavailable()
    public static let v11 = Self(for: .v11)
    public static let v12 = Self(for: .v12)
    public static let v13 = Self(for: .v13)
    public static let v14 = Self(for: .v14)
    public static let v15 = Self(for: .v15)
}

extension visionOSViewVersion<MapType, MKMapView> {
    public static let v1 = Self(for: .v1)
    public static let v2 = Self(for: .v2)
}
#endif
#endif
