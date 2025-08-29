//
//  DeviceInfoHelper.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/10.
//

import Foundation
import UIKit

/// Helper class for retrieving device information including model identification
/// and system details. Provides device-specific information for benchmark results.
class DeviceInfoHelper {
    static let shared = DeviceInfoHelper()
    
    private init() {}
    
    /// Gets the device model identifier (e.g., "iPhone14,7")
    func getDeviceIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        
        let machineMirror = Mirror(reflecting: systemInfo.machine)
        let identifier = machineMirror.children.reduce("") { identifier, element in
            guard let value = element.value as? Int8, value != 0 else { return identifier }
            return identifier + String(UnicodeScalar(UInt8(value)))
        }
        
        return identifier
    }
    
    /// Gets the user-friendly device name (e.g., "iPhone 13 mini")
    func getDeviceModelName() -> String {
        let identifier = getDeviceIdentifier()
        return mapIdentifierToModelName(identifier)
    }
    
    /// Gets detailed device information including model and system version
    func getDeviceInfo() -> String {
        let device = UIDevice.current
        let systemVersion = device.systemVersion
        let modelName = getDeviceModelName()
        return "\(modelName), iOS \(systemVersion)"
    }
    
    private func mapIdentifierToModelName(_ identifier: String) -> String {
        // iPhone mappings
        let iPhoneMappings: [String: String] = [
            // iPhone 13 series
            "iPhone14,4": "iPhone 13 mini",
            "iPhone14,5": "iPhone 13",
            "iPhone14,2": "iPhone 13 Pro",
            "iPhone14,3": "iPhone 13 Pro Max",
            
            // iPhone 14 series
            "iPhone14,7": "iPhone 14",
            "iPhone14,8": "iPhone 14 Plus",
            "iPhone15,2": "iPhone 14 Pro",
            "iPhone15,3": "iPhone 14 Pro Max",
            
            // iPhone 15 series
            "iPhone15,4": "iPhone 15",
            "iPhone15,5": "iPhone 15 Plus",
            "iPhone16,1": "iPhone 15 Pro",
            "iPhone16,2": "iPhone 15 Pro Max",
            
            // iPhone 16 series
            "iPhone17,1": "iPhone 16",
            "iPhone17,2": "iPhone 16 Plus",
            "iPhone17,3": "iPhone 16 Pro",
            "iPhone17,4": "iPhone 16 Pro Max",
            
            // iPhone SE series
            "iPhone12,8": "iPhone SE (2nd generation)",
            "iPhone14,6": "iPhone SE (3rd generation)",
            
            // Older iPhones
            "iPhone13,1": "iPhone 12 mini",
            "iPhone13,2": "iPhone 12",
            "iPhone13,3": "iPhone 12 Pro",
            "iPhone13,4": "iPhone 12 Pro Max",
            "iPhone12,1": "iPhone 11",
            "iPhone12,3": "iPhone 11 Pro",
            "iPhone12,5": "iPhone 11 Pro Max",
        ]
        
        // iPad mappings
        let iPadMappings: [String: String] = [
            // iPad Pro 12.9-inch
            "iPad13,8": "iPad Pro (12.9-inch) (5th generation)",
            "iPad13,9": "iPad Pro (12.9-inch) (5th generation)",
            "iPad13,10": "iPad Pro (12.9-inch) (5th generation)",
            "iPad13,11": "iPad Pro (12.9-inch) (5th generation)",
            "iPad14,5": "iPad Pro (12.9-inch) (6th generation)",
            "iPad14,6": "iPad Pro (12.9-inch) (6th generation)",
            
            // iPad Pro 11-inch
            "iPad13,4": "iPad Pro (11-inch) (3rd generation)",
            "iPad13,5": "iPad Pro (11-inch) (3rd generation)",
            "iPad13,6": "iPad Pro (11-inch) (3rd generation)",
            "iPad13,7": "iPad Pro (11-inch) (3rd generation)",
            "iPad14,3": "iPad Pro (11-inch) (4th generation)",
            "iPad14,4": "iPad Pro (11-inch) (4th generation)",
            
            // iPad Air
            "iPad13,1": "iPad Air (4th generation)",
            "iPad13,2": "iPad Air (4th generation)",
            "iPad13,16": "iPad Air (5th generation)",
            "iPad13,17": "iPad Air (5th generation)",
            
            // iPad mini
            "iPad14,1": "iPad mini (6th generation)",
            "iPad14,2": "iPad mini (6th generation)",
            
            // iPad (regular)
            "iPad12,1": "iPad (9th generation)",
            "iPad12,2": "iPad (9th generation)",
            "iPad13,18": "iPad (10th generation)",
            "iPad13,19": "iPad (10th generation)",
        ]
        
        // Try iPhone mappings first
        if let modelName = iPhoneMappings[identifier] {
            return modelName
        }
        
        // Try iPad mappings
        if let modelName = iPadMappings[identifier] {
            return modelName
        }
        
        // Check for simulator
        if identifier == "x86_64" || identifier == "i386" {
            return "Simulator"
        }
        
        // Return raw identifier if no mapping found
        return identifier
    }
}
