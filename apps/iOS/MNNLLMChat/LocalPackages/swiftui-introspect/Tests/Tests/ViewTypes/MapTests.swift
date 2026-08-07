#if canImport(MapKit)
import MapKit
import SwiftUI
import SwiftUIIntrospect
import XCTest

@available(iOS 14, tvOS 14, macOS 11, *)
@MainActor
final class MapTests: XCTestCase {
    typealias PlatformMap = MKMapView

    func testMap() throws {
        guard #available(iOS 14, tvOS 14, macOS 11, *) else {
            throw XCTSkip()
        }

        XCTAssertViewIntrospection(of: PlatformMap.self) { spies in
            let spy0 = spies[0]
            let spy1 = spies[1]
            let spy2 = spies[2]

            let region = Binding.constant(MKCoordinateRegion(center: CLLocationCoordinate2D(latitude: 51.507222, longitude: -0.1275), span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5)))

            VStack {
                Map(coordinateRegion: region)
                    .introspect(
                        .map,
                        on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .macOS(.v11, .v12, .v13, .v14), .visionOS(.v1, .v2),
                        customize: spy0
                    )

                Map(coordinateRegion: region)
                    .introspect(
                        .map,
                        on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .macOS(.v11, .v12, .v13, .v14), .visionOS(.v1, .v2),
                        customize: spy1
                    )

                Map(coordinateRegion: region)
                    .introspect(
                        .map,
                        on: .iOS(.v14, .v15, .v16, .v17, .v18), .tvOS(.v14, .v15, .v16, .v17, .v18), .macOS(.v11, .v12, .v13, .v14), .visionOS(.v1, .v2),
                        customize: spy2
                    )
            }
        } extraAssertions: {
            XCTAssertNotIdentical($0[safe: 0], $0[safe: 1])
            XCTAssertNotIdentical($0[safe: 0], $0[safe: 2])
            XCTAssertNotIdentical($0[safe: 1], $0[safe: 2])
        }
    }
}
#endif
