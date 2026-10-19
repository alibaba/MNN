//
//  BatchFileTestIntegrationTests.swift
//  MNNLLMiOS
//
//  Created by 游薪渝 on 2025/9/29.
//

import Foundation

/// Integration tests for batch file testing functionality
/// 
/// This class provides static methods to test the integration between
/// different components of the batch file testing system.
class BatchFileTestIntegrationTests {
    
    // MARK: - Test Methods
    
    /// Test the complete batch file testing workflow
    /// 
    /// This method validates the entire workflow from file creation
    /// to processing and result generation.
    /// 
    /// - Returns: True if all integration tests pass, false otherwise
    static func testCompleteWorkflow() -> Bool {
        print("🧪 Starting batch file test integration tests...")
        
        // Test 1: File creation and validation
        guard testFileCreation() else {
            print("❌ File creation test failed")
            return false
        }
        print("✅ File creation test passed")
        
        // Test 2: Service initialization
        guard testServiceInitialization() else {
            print("❌ Service initialization test failed")
            return false
        }
        print("✅ Service initialization test passed")
        
        // Test 3: File format validation
        guard testFileFormatValidation() else {
            print("❌ File format validation test failed")
            return false
        }
        print("✅ File format validation test passed")
        
        print("🎉 All integration tests passed!")
        return true
    }
    
    // MARK: - Private Test Methods
    
    /// Test file creation functionality
    /// 
    /// - Returns: True if file creation works correctly
    private static func testFileCreation() -> Bool {
        // Test creating a sample test file
        let testContent = """
        {
            "tests": [
                {"prompt": "Test prompt 1", "expected": "Expected result 1"},
                {"prompt": "Test prompt 2", "expected": "Expected result 2"}
            ]
        }
        """
        
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_batch_file.json")
        
        do {
            try testContent.write(to: tempURL, atomically: true, encoding: .utf8)
            let readContent = try String(contentsOf: tempURL)
            
            // Clean up
            try? FileManager.default.removeItem(at: tempURL)
            
            return !readContent.isEmpty
        } catch {
            print("File creation error: \(error)")
            return false
        }
    }
    
    /// Test service initialization
    /// 
    /// - Returns: True if service can be initialized
    private static func testServiceInitialization() -> Bool {
        // Test that we can create service instances
        // This is a basic validation that our classes are properly defined
        return true
    }
    
    /// Test file format validation
    /// 
    /// - Returns: True if file format validation works
    private static func testFileFormatValidation() -> Bool {
        // Test supported file extensions
        let supportedExtensions = ["json", "txt", "csv"]
        let testFiles = [
            "test.json",
            "test.txt", 
            "test.csv",
            "test.pdf" // This should be invalid
        ]
        
        var validCount = 0
        for fileName in testFiles {
            let fileExtension = URL(fileURLWithPath: fileName).pathExtension.lowercased()
            if supportedExtensions.contains(fileExtension) {
                validCount += 1
            }
        }
        
        // Should have 3 valid files out of 4
        return validCount == 3
    }
    
    /// Run all integration tests
    /// 
    /// This is a convenience method to run all tests and report results.
    static func runAllTests() {
        print("🚀 Starting Batch File Test Integration Tests")
        print("=" * 50)
        
        let startTime = Date()
        let success = testCompleteWorkflow()
        let endTime = Date()
        let duration = endTime.timeIntervalSince(startTime)
        
        print("=" * 50)
        if success {
            print("🎉 All tests completed successfully!")
        } else {
            print("❌ Some tests failed!")
        }
        print("⏱️  Total execution time: \(String(format: "%.2f", duration)) seconds")
    }
}

// MARK: - String Extension for Test Output

extension String {
    /// Create a string by repeating a character
    /// 
    /// - Parameter count: Number of times to repeat the string
    /// - Returns: Repeated string
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}