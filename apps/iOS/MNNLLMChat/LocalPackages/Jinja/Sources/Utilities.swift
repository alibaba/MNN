//
//  Utilities.swift
//
//
//  Created by John Mai on 2024/3/20.
//

import Foundation

func range(start: Int, stop: Int? = nil, step: Int = 1) -> [Int] {
    let stopUnwrapped = stop ?? start
    let startValue = stop == nil ? 0 : start
    let stopValue = stop == nil ? start : stopUnwrapped

    return stride(from: startValue, to: stopValue, by: step).map { $0 }
}

func slice<T>(_ array: [T], start: Int? = nil, stop: Int? = nil, step: Int? = 1) -> [T] {
    let arrayCount = array.count
    let startValue = start ?? 0
    let stopValue = stop ?? arrayCount
    let step = step ?? 1
    var slicedArray = [T]()
    if step > 0 {
        let startIndex = startValue < 0 ? max(arrayCount + startValue, 0) : min(startValue, arrayCount)
        let stopIndex = stopValue < 0 ? max(arrayCount + stopValue, 0) : min(stopValue, arrayCount)
        for i in stride(from: startIndex, to: stopIndex, by: step) {
            slicedArray.append(array[i])
        }
    } else {
        let startIndex = startValue < 0 ? max(arrayCount + startValue, -1) : min(startValue, arrayCount - 1)
        let stopIndex = stopValue < -1 ? max(arrayCount + stopValue, -1) : min(stopValue, arrayCount - 1)
        for i in stride(from: startIndex, through: stopIndex, by: step) {
            slicedArray.append(array[i])
        }
    }
    return slicedArray
}

func toJSON(_ input: any RuntimeValue, indent: Int? = nil, depth: Int = 0, whitespaceControl: Bool = false) throws
    -> String
{
    // If whitespaceControl is true, output compact JSON
    if whitespaceControl {
        switch input {
        case is NullValue, is UndefinedValue:
            return "null"
        case let value as NumericValue:
            return String(describing: value.value)
        case let value as StringValue:
            let escapedValue = value.value
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
                .replacingOccurrences(of: "\t", with: "\\t")
            return "\"\(escapedValue)\""
        case let value as BooleanValue:
            return value.value ? "true" : "false"
        case let arr as ArrayValue:
            let elements = try arr.value.map {
                try toJSON($0, indent: nil, depth: 0, whitespaceControl: true)
            }
            return "[\(elements.joined(separator: ", "))]"
        case let obj as ObjectValue:
            let pairs = try obj.orderedKeys.map { key in
                guard let value = obj.value[key] else {
                    throw JinjaError.runtime("Missing value for key: \(key)")
                }
                let jsonValue = try toJSON(value, indent: nil, depth: 0, whitespaceControl: true)
                return "\"\(key)\": \(jsonValue)"
            }
            return "{\(pairs.joined(separator: ", "))}"
        default:
            throw JinjaError.runtime("Cannot convert to JSON: \(type(of: input))")
        }
    }
    let currentDepth = depth
    let indentValue = indent != nil ? String(repeating: " ", count: indent!) : ""
    let basePadding = indent != nil ? "\n" + String(repeating: indentValue, count: currentDepth) : ""
    let childrenPadding = indent != nil ? basePadding + indentValue : ""
    switch input {
    case is NullValue, is UndefinedValue:
        return "null"
    case let value as NumericValue:
        return String(describing: value.value)
    case let value as StringValue:
        // Properly escape special characters for JSON strings
        let escapedValue = value.value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
        return "\"\(escapedValue)\""
    case let value as BooleanValue:
        return value.value ? "true" : "false"
    case let arr as ArrayValue:
        let core = try arr.value.map {
            try toJSON($0, indent: indent, depth: currentDepth + 1, whitespaceControl: whitespaceControl)
        }
        if indent != nil && !whitespaceControl {
            return "[\(childrenPadding)\(core.joined(separator: ",\(childrenPadding)"))\(basePadding)]"
        } else {
            return "[\(core.joined(separator: ", "))]"
        }
    case let obj as ObjectValue:
        // Use orderedKeys to maintain insertion order
        let pairs = try obj.orderedKeys.map { key in
            guard let value = obj.value[key] else {
                throw JinjaError.runtime("Missing value for key: \(key)")
            }
            let jsonValue = try toJSON(
                value,
                indent: indent,
                depth: currentDepth + 1,
                whitespaceControl: whitespaceControl
            )
            return "\"\(key)\": \(jsonValue)"
        }
        if indent != nil && !whitespaceControl {
            return "{\(childrenPadding)\(pairs.joined(separator: ",\(childrenPadding)"))\(basePadding)}"
        } else {
            return "{\(pairs.joined(separator: ", "))}"
        }
    default:
        throw JinjaError.runtime("Cannot convert to JSON: \(type(of: input))")
    }
}

// Helper function to convert values to JSON strings
func jsonString(_ value: Any) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: value)
    guard let string = String(data: data, encoding: .utf8) else {
        throw JinjaError.runtime("Failed to convert value to JSON string")
    }
    return string
}

extension String {
    func titleCase() -> String {
        return self.components(separatedBy: .whitespacesAndNewlines)
            .map { word in
                guard let firstChar = word.first else { return "" }
                return String(firstChar).uppercased() + word.dropFirst()
            }
            .joined(separator: " ")
    }

    func indent(_ width: Int, first: Bool = false, blank: Bool = false) -> String {
        let indentString = String(repeating: " ", count: width)
        return self.components(separatedBy: .newlines)
            .enumerated()
            .map { (index, line) in
                if line.isEmpty && !blank {
                    return line
                }
                if index == 0 && !first {
                    return line
                }
                return indentString + line
            }
            .joined(separator: "\n")
    }
}

func stringify(_ value: any RuntimeValue, indent: Int = 4, whitespaceControl: Bool = false) throws -> String {
    if let stringValue = value as? StringValue {
        return "\"\(stringValue.value)\""
    } else if let numericValue = value as? NumericValue {
        return String(describing: numericValue.value)
    } else if let booleanValue = value as? BooleanValue {
        return booleanValue.value ? "true" : "false"
    } else if let objectValue = value as? ObjectValue {
        return try toJSON(objectValue, indent: indent, whitespaceControl: whitespaceControl)
    } else if let arrayValue = value as? ArrayValue {
        return try toJSON(arrayValue, indent: indent, whitespaceControl: whitespaceControl)
    } else if value is NullValue {
        return "null"
    } else if value is UndefinedValue {
        return "undefined"
    } else {
        return ""
    }
}

func areEqual(_ left: any RuntimeValue, _ right: any RuntimeValue) throws -> Bool {
    if let leftObj = left as? ObjectValue, let rightObj = right as? ObjectValue {
        // Compare ObjectValues by their contents
        guard leftObj.storage.keys == rightObj.storage.keys else {
            return false
        }

        for key in leftObj.storage.keys {
            guard let leftValue = leftObj.storage[key],
                let rightValue = rightObj.storage[key],
                try areEqual(leftValue, rightValue)
            else {
                return false
            }
        }
        return true
    } else if let leftStr = left as? StringValue, let rightStr = right as? StringValue {
        return leftStr.value == rightStr.value
    } else if let leftNum = left as? NumericValue, let rightNum = right as? NumericValue {
        if let leftInt = leftNum.value as? Int, let rightInt = rightNum.value as? Int {
            return leftInt == rightInt
        } else if let leftDouble = leftNum.value as? Double, let rightDouble = rightNum.value as? Double {
            return leftDouble == rightDouble
        }
    } else if let leftArr = left as? ArrayValue, let rightArr = right as? ArrayValue {
        guard leftArr.value.count == rightArr.value.count else {
            return false
        }
        for (leftItem, rightItem) in zip(leftArr.value, rightArr.value) {
            guard try areEqual(leftItem, rightItem) else {
                return false
            }
        }
        return true
    } else if left is NullValue && right is NullValue {
        return true
    } else if left is UndefinedValue && right is UndefinedValue {
        return true
    } else if let leftBool = left as? BooleanValue, let rightBool = right as? BooleanValue {
        return leftBool.value == rightBool.value
    }
    // If types don't match, return false
    return false
}
