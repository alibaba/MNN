import SnapshotTesting
import XCTest

final class StatusBarStyleUITests: UITestCase {
    override var testCase: TestCase {
        .statusBarStyle
    }

    func test() throws {
        guard #unavailable(iOS 17) else {
            throw XCTSkip("SimulatorStatusMagic stopped working in iOS 17, so we can no longer consistently compare status bar screenshots")
        }

        app.buttons["Navigate To Detail"].tap()
        app.buttons["Navigate To Detail"].tap()
        app.buttons["Navigate Back"].tap()

        let iOSDevice = UIDevice.current.userInterfaceIdiom == .pad ? "ipad" : "iphone"
        let iOSVersion = ProcessInfo().operatingSystemVersion
        func screenshotName(_ number: Int) -> String {
            "\(iOSDevice)-ios-\(iOSVersion.majorVersion)-screenshot-\(number)"
        }

        assertSnapshot(
            matching: app.windows.firstMatch.screenshot().image,
            as: .image(perceptualPrecision: 0.95),
            named: screenshotName(1)
        )
    }
}
