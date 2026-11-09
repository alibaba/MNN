import SimulatorStatusMagic
import XCTest

class UITestCase: XCTestCase {
    var testCase: TestCase {
        preconditionFailure("Please override this property")
    }
    
    let app = XCUIApplication()
    
    override func invokeTest() {
        SDStatusBarManager.sharedInstance().enableOverrides()
        
        continueAfterFailure = false
        
        app.launchEnvironment["testCase"] = testCase.rawValue
        app.launch()
        
        super.invokeTest()
    }
}
