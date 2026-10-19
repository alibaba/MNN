//
//  Environment.swift
//
//
//  Created by John Mai on 2024/3/23.
//

import Foundation
import OrderedCollections

class Environment {
    var parent: Environment?

    var variables: [String: any RuntimeValue] = [
        "namespace": FunctionValue(value: { args, _ in
            if args.isEmpty {
                return ObjectValue(value: [:])
            }
            guard args.count == 1, let objectArg = args[0] as? ObjectValue else {
                throw JinjaError.runtime("`namespace` expects either zero arguments or a single object argument")
            }
            return objectArg
        }),

        // Add strftime_now function to handle date formatting in templates
        "strftime_now": FunctionValue(value: { args, _ in
            let now = Date()

            if args.count > 0, let formatArg = args[0] as? StringValue {
                let format = formatArg.value

                let result = formatDate(now, withFormat: format)
                return StringValue(value: result)
            }

            // Default format if no arguments
            let formatter = DateFormatter()
            formatter.dateFormat = "MMMM dd, yyyy"
            return StringValue(value: formatter.string(from: now))
        }),
    ]

    lazy var tests: [String: ([any RuntimeValue]) throws -> Bool] = [
        "odd": { args in
            if let arg = args.first as? NumericValue, let intValue = arg.value as? Int {
                return intValue % 2 != 0
            } else {
                throw JinjaError.runtime(
                    "Cannot apply test 'odd' to type: \(type(of: args.first)) with value \(String(describing: args.first?.value))"
                )
            }
        },
        "even": { args in
            if let arg = args.first as? NumericValue, let intValue = arg.value as? Int {
                return intValue % 2 == 0
            } else {
                throw JinjaError.runtime(
                    "Cannot apply test 'even' to type: \(type(of: args.first)) with value \(String(describing: args.first?.value))"
                )
            }
        },
        "divisibleby": { args in
            guard let value = args[0] as? NumericValue,
                let num = args[1] as? NumericValue,
                let intValue = value.value as? Int,
                let intNum = num.value as? Int
            else {
                throw JinjaError.runtime("divisibleby test requires two integers")
            }
            return intValue % intNum == 0
        },
        "defined": { args in
            return !(args[0] is UndefinedValue)
        },
        "undefined": { args in
            return args[0] is UndefinedValue
        },
        "filter": { [weak self] (args: [any RuntimeValue]) throws -> Bool in
            guard let name = args[0] as? StringValue else {
                throw JinjaError.runtime("filter test requires a string")
            }
            return self?.filters.keys.contains(name.value) ?? false
        },
        "test": { [weak self] (args: [any RuntimeValue]) throws -> Bool in
            guard let name = args[0] as? StringValue else {
                throw JinjaError.runtime("test test requires a string")
            }
            return self?.tests.keys.contains(name.value) ?? false
        },
        "none": { args in
            return args[0] is NullValue
        },
        "boolean": { args in
            return args[0] is BooleanValue
        },
        "false": { args in
            if let arg = args[0] as? BooleanValue {
                return !arg.value
            }
            return false
        },
        "true": { args in
            if let arg = args[0] as? BooleanValue {
                return arg.value
            }
            return false
        },
        "integer": { args in
            if let arg = args[0] as? NumericValue {
                return arg.value is Int
            }
            return false
        },
        "float": { args in
            if let numericValue = args[0] as? NumericValue {
                return numericValue.value is Double
            }
            return false
        },
        "lower": { args in
            if let arg = args[0] as? StringValue {
                return arg.value == arg.value.lowercased()
            }
            return false
        },
        "upper": { args in
            if let arg = args[0] as? StringValue {
                return arg.value == arg.value.uppercased()
            }
            return false
        },
        "string": { args in
            return args[0] is StringValue
        },
        "mapping": { args in
            return args[0] is ObjectValue
        },
        "number": { args in
            return args[0] is NumericValue
        },
        "sequence": { args in
            let value = args[0]
            if value is ArrayValue || value is StringValue {
                return true
            }
            return false
        },
        "iterable": { args in
            return args[0] is ArrayValue || args[0] is StringValue || args[0] is ObjectValue
        },
        "callable": { args in
            return args[0] is FunctionValue
        },
        // TODO: Implement "sameas"
        // TODO: Implement "escaped"
        "in": { [unowned self] args in
            guard let seq = args[1] as? ArrayValue else {
                throw JinjaError.runtime("in test requires a sequence")
            }
            return seq.value.contains { item in
                self.doEqualTo([args[0], item])
            }
        },
        "==": { [unowned self] args in self.doEqualTo(args) },
        "eq": { [unowned self] args in self.doEqualTo(args) },
        "equalto": { [unowned self] args in self.doEqualTo(args) },
        "!=": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("!= test requires two arguments")
            }
            return !self.doEqualTo(args)
        },
        "ne": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("ne test requires two arguments")
            }
            return !self.doEqualTo(args)
        },
        ">": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("> test requires two arguments")
            }
            return try self.doGreaterThan(args)
        },
        "gt": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("gt test requires two arguments")
            }
            return try self.doGreaterThan(args)
        },
        "greaterthan": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("greaterthan test requires two arguments")
            }
            return try self.doGreaterThan(args)
        },
        ">=": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime(">= test requires two arguments")
            }
            return try self.doGreaterThanOrEqual(args)
        },
        "ge": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("ge test requires two arguments")
            }
            return try self.doGreaterThanOrEqual(args)
        },
        "<": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("< test requires two arguments")
            }
            return try self.doLessThan(args)
        },
        "lt": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("lt test requires two arguments")
            }
            return try self.doLessThan(args)
        },
        "lessthan": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("lessthan test requires two arguments")
            }
            return try self.doLessThan(args)
        },
        "<=": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("<= test requires two arguments")
            }
            return try self.doLessThanOrEqual(args)
        },
        "le": { [unowned self] args in
            guard args.count == 2 else {
                throw JinjaError.runtime("le test requires two arguments")
            }
            return try self.doLessThanOrEqual(args)
        },
    ]

    lazy var filters: [String: ([any RuntimeValue], Environment) throws -> any RuntimeValue] = [
        "abs": { args, env in
            guard args.count == 1 else {
                throw JinjaError.runtime("abs filter requires exactly one argument, but \(args.count) were provided")
            }
            guard let numericValue = args[0] as? NumericValue else {
                throw JinjaError.runtime("abs filter requires a number")
            }
            if let intValue = numericValue.value as? Int {
                let absValue = abs(intValue)
                return NumericValue(value: absValue)
            } else if let doubleValue = numericValue.value as? Double {
                let absValue = abs(doubleValue)
                return NumericValue(value: absValue)
            } else {
                throw JinjaError.runtime("Unsupported numeric type for abs filter")
            }
        },
        "attr": { args, env in
            guard args.count >= 2 else {
                throw JinjaError.runtime("attr filter requires an object and attribute name")
            }
            let obj = args[0]
            // Convert name to string (similar to str(name) in Python)
            let name: String
            if let stringValue = args[1] as? StringValue {
                name = stringValue.value
            } else {
                // Try to convert the name to string
                do {
                    name = try stringify(args[1])
                } catch {
                    return UndefinedValue()
                }
            }
            // Handle different object types
            if let objectValue = obj as? ObjectValue {
                // Return the raw value if it exists
                if let value = objectValue.value[name] {
                    return value
                }
            }
            // If attribute is not found, return undefined
            return UndefinedValue()
        },
        "batch": { args, env in
            guard let arrayValue = args[0] as? ArrayValue,
                let linecount = args[1] as? NumericValue,
                let count = linecount.value as? Int
            else {
                throw JinjaError.runtime("batch filter requires an array and line count")
            }
            let fillWith = args.count > 2 ? args[2] : nil
            var result: [[any RuntimeValue]] = []
            var temp: [any RuntimeValue] = []
            for item in arrayValue.value {
                if temp.count == count {
                    result.append(temp)
                    temp = []
                }
                temp.append(item)
            }
            if !temp.isEmpty {
                if let fill = fillWith, temp.count < count {
                    temp += Array(repeating: fill, count: count - temp.count)
                }
                result.append(temp)
            }
            return ArrayValue(value: result.map { ArrayValue(value: $0) })
        },
        "capitalize": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("capitalize filter requires a string")
            }
            let str = stringValue.value
            guard let firstChar = str.first else {
                return stringValue  // Empty string, return as is
            }
            return StringValue(value: String(firstChar).uppercased() + str.dropFirst().lowercased())
        },
        "center": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("center filter requires a string")
            }
            let width = (args.count > 1 && args[1] is NumericValue) ? (args[1] as! NumericValue).value as! Int : 80
            let str = stringValue.value

            // If string is longer than width, return original string
            if str.count >= width {
                return stringValue
            }

            // Calculate total padding needed
            let padding = width - str.count

            // Calculate left and right padding
            // When padding is odd, the extra space goes to the right
            let leftPadding = padding / 2
            let rightPadding = padding - leftPadding  // This ensures extra padding goes to the right

            // Create padded string
            return StringValue(
                value: String(repeating: " ", count: leftPadding) + str + String(repeating: " ", count: rightPadding)
            )
        },
        "count": { args, env in
            let value = args[0]
            if let arrayValue = value as? ArrayValue {
                return NumericValue(value: arrayValue.value.count)
            } else if let stringValue = value as? StringValue {
                return NumericValue(value: stringValue.value.count)
            } else if let objectValue = value as? ObjectValue {
                return NumericValue(value: objectValue.value.count)
            }
            throw JinjaError.runtime("Cannot count value of type \(type(of: value))")
        },
        "d": { [unowned self] args, env in try self.doDefault(args, env) },
        "default": { [unowned self] args, env in try self.doDefault(args, env) },
        "dictsort": { args, env in
            guard let dict = args[0] as? ObjectValue else {
                throw JinjaError.runtime("dictsort filter requires a dictionary")
            }
            let caseSensitive = args.count > 1 ? (args[1] as? BooleanValue)?.value ?? false : false
            let by = args.count > 2 ? (args[2] as? StringValue)?.value ?? "key" : "key"
            let reverse = args.count > 3 ? (args[3] as? BooleanValue)?.value ?? false : false
            let sortedDict = try dict.storage.sorted { (item1, item2) in
                let a: Any, b: Any
                if by == "key" {
                    a = item1.key
                    b = item2.key
                } else if by == "value" {
                    a = item1.value
                    b = item2.value
                } else {
                    throw JinjaError.runtime("Invalid 'by' argument for dictsort filter")
                }
                let result: Bool
                if let aString = a as? String, let bString = b as? String {
                    result = caseSensitive ? aString < bString : aString.lowercased() < bString.lowercased()
                } else if let aNumeric = a as? NumericValue, let bNumeric = b as? NumericValue {
                    if let aInt = aNumeric.value as? Int, let bInt = bNumeric.value as? Int {
                        result = aInt < bInt
                    } else if let aDouble = aNumeric.value as? Double, let bDouble = bNumeric.value as? Double {
                        result = aDouble < bDouble
                    } else {
                        throw JinjaError.runtime("Cannot compare values in dictsort filter")
                    }
                } else {
                    throw JinjaError.runtime("Cannot compare values in dictsort filter")
                }
                return reverse ? !result : result
            }
            return ArrayValue(
                value: sortedDict.map { (key, value) in
                    return ArrayValue(value: [StringValue(value: key), value])
                }
            )
        },
        "e": { [unowned self] args, env in try self.doEscape(args, env) },
        "escape": { [unowned self] args, env in try self.doEscape(args, env) },
        "filesizeformat": { args, env in
            guard let value = args[0] as? NumericValue else {
                throw JinjaError.runtime("filesizeformat filter requires a numeric value")
            }

            let size: Double
            if let intValue = value.value as? Int {
                size = Double(intValue)
            } else if let doubleValue = value.value as? Double {
                size = doubleValue
            } else {
                throw JinjaError.runtime("filesizeformat filter requires a numeric value")
            }

            let binary = args.count > 1 ? (args[1] as? BooleanValue)?.value ?? false : false
            let units =
                binary
                ? [" Bytes", " KiB", " MiB", " GiB", " TiB", " PiB", " EiB", " ZiB", " YiB"]
                : [" Bytes", " kB", " MB", " GB", " TB", " PB", " EB", " ZB", " YB"]
            let base: Double = binary ? 1024.0 : 1000.0

            if size < base {
                return StringValue(value: "\(Int(size)) Bytes")
            }

            let exp = Int(log(size) / log(base))
            let unit = units[min(exp, units.count - 1)]
            let num = size / pow(base, Double(exp))
            return StringValue(value: String(format: "%.1f%@", num, unit))
        },
        "first": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("first filter requires an array")
            }
            return arrayValue.value.first ?? UndefinedValue()
        },
        "float": { args, env in
            guard let value = args[0] as? NumericValue else {
                return NumericValue(value: 0.0)
            }
            if let doubleValue = value.value as? Double {
                return NumericValue(value: doubleValue)
            } else if let intValue = value.value as? Int {
                return NumericValue(value: Double(intValue))
            } else {
                return NumericValue(value: 0.0)
            }
        },
        "forceescape": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("forceescape filter requires a string")
            }
            return StringValue(
                value: stringValue.value.replacingOccurrences(of: "&", with: "&amp;")
                    .replacingOccurrences(of: "<", with: "&lt;")
                    .replacingOccurrences(of: ">", with: "&gt;")
                    .replacingOccurrences(of: "\"", with: "&quot;")
                    .replacingOccurrences(of: "'", with: "&#39;")
            )
        },
        "format": { args, env in
            guard let format = args[0] as? StringValue else {
                throw JinjaError.runtime("format filter requires a format string")
            }
            // Get the values after the format string
            let formatArgs = Array(args.dropFirst())
            // Convert the values to strings
            let formatValues = formatArgs.map { arg -> String in
                if let stringValue = arg as? StringValue {
                    return stringValue.value
                } else if let numericValue = arg as? NumericValue {
                    if let intValue = numericValue.value as? Int {
                        return String(intValue)
                    } else if let doubleValue = numericValue.value as? Double {
                        return String(doubleValue)
                    }
                }
                return String(describing: arg)
            }
            // Replace %s with values one by one
            var result = format.value
            for value in formatValues {
                if let range = result.range(of: "%s") {
                    result.replaceSubrange(range, with: value)
                } else if let range = result.range(of: "%d") {
                    result.replaceSubrange(range, with: value)
                }
            }
            return StringValue(value: result)
        },
        "groupby": { [unowned self] args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("groupby filter requires an array")
            }
            guard let attribute = args[1] as? StringValue else {
                throw JinjaError.runtime("groupby filter requires an attribute name")
            }
            let defaultValue = args.count > 2 ? args[2] : nil
            let caseSensitive = args.count > 3 ? (args[3] as? BooleanValue)?.value ?? false : false

            // Helper function to get nested attribute value
            func getAttributeValue(_ obj: ObjectValue, _ path: String) -> any RuntimeValue {
                let components = path.split(separator: ".")
                var current: any RuntimeValue = obj

                for component in components {
                    if let currentObj = current as? ObjectValue,
                        let value = currentObj.value[String(component)]
                    {
                        current = value
                    } else {
                        return defaultValue ?? UndefinedValue()
                    }
                }
                return current
            }

            // Sort the array first
            let sorted = arrayValue.value.sorted { (a, b) in
                guard let aObj = a as? ObjectValue,
                    let bObj = b as? ObjectValue
                else {
                    return false
                }

                let aValue = getAttributeValue(aObj, attribute.value)
                let bValue = getAttributeValue(bObj, attribute.value)

                if let aStr = aValue as? StringValue,
                    let bStr = bValue as? StringValue
                {
                    let aCompare = caseSensitive ? aStr.value : aStr.value.lowercased()
                    let bCompare = caseSensitive ? bStr.value : bStr.value.lowercased()
                    return aCompare < bCompare
                }
                // Add other comparison types as needed
                return false
            }

            // Group the sorted array
            var groups: [(grouper: any RuntimeValue, list: [any RuntimeValue])] = []
            var currentGroup: [any RuntimeValue] = []
            var currentKey: (any RuntimeValue)? = nil  // Changed to var and explicitly initialized as nil

            for item in sorted {
                guard let obj = item as? ObjectValue else { continue }
                let value = getAttributeValue(obj, attribute.value)
                let key =
                    caseSensitive
                    ? value : (value as? StringValue).map { StringValue(value: $0.value.lowercased()) } ?? value

                if let existingKey = currentKey {  // Changed to different name for binding
                    if self.doEqualTo([key, existingKey]) {
                        currentGroup.append(item)
                    } else {
                        if !currentGroup.isEmpty {
                            // Use the first item's actual value as the grouper
                            if let firstObj = currentGroup[0] as? ObjectValue {
                                let grouper = getAttributeValue(firstObj, attribute.value)
                                groups.append((grouper: grouper, list: currentGroup))
                            }
                        }
                        currentGroup = [item]
                        currentKey = key  // Now works because currentKey is var
                    }
                } else {
                    currentGroup = [item]
                    currentKey = key
                }
            }

            // Add the last group
            if !currentGroup.isEmpty {
                if let firstObj = currentGroup[0] as? ObjectValue {
                    let grouper = getAttributeValue(firstObj, attribute.value)
                    groups.append((grouper: grouper, list: currentGroup))
                }
            }

            // Convert groups to array of objects with 'grouper' and 'list' keys
            return ArrayValue(
                value: groups.map { group in
                    ObjectValue(value: [
                        "grouper": group.grouper,
                        "list": ArrayValue(value: group.list),
                    ])
                }
            )
        },
        "indent": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("indent filter requires a string")
            }
            // Determine indentation width
            var indent: String
            if args.count > 1 {
                if let width = args[1] as? NumericValue, let intWidth = width.value as? Int {
                    indent = String(repeating: " ", count: intWidth)
                } else if let stringWidth = args[1] as? StringValue {
                    indent = stringWidth.value
                } else {
                    indent = String(repeating: " ", count: 4)  // Default
                }
            } else {
                indent = String(repeating: " ", count: 4)  // Default
            }
            let first = args.count > 2 ? (args[2] as? BooleanValue)?.value ?? false : false
            let blank = args.count > 3 ? (args[3] as? BooleanValue)?.value ?? false : false
            // Add a newline to the end of the string (Python quirk)
            let modifiedStringValue = stringValue.value + "\n"
            // Split into lines
            var lines = modifiedStringValue.components(separatedBy: "\n")
            // Remove the last line (which is always empty due to the added newline)
            lines.removeLast()
            if lines.isEmpty {
                return StringValue(value: "")
            }
            var result: String
            // Handle first line
            if first {
                result = indent + lines[0]
            } else {
                result = lines[0]
            }
            // Process remaining lines
            if lines.count > 1 {
                let remainingLines = lines.dropFirst().map { line -> String in
                    if line.isEmpty {
                        return blank ? indent + line : line
                    } else {
                        return indent + line
                    }
                }
                result += "\n" + remainingLines.joined(separator: "\n")
            }
            return StringValue(value: result)
        },
        "int": { args, env in
            if let numericValue = args[0] as? NumericValue {
                if let intValue = numericValue.value as? Int {
                    return NumericValue(value: intValue)
                } else if let doubleValue = numericValue.value as? Double {
                    return NumericValue(value: Int(doubleValue))
                }
            } else if let stringValue = args[0] as? StringValue {
                if let intValue = Int(stringValue.value) {
                    return NumericValue(value: intValue)
                } else if let doubleValue = Double(stringValue.value) {
                    return NumericValue(value: Int(doubleValue))
                }
            }
            // Return 0 for any other case (including invalid strings)
            return NumericValue(value: 0)
        },
        "items": { args, env in
            guard let value = args.first else {
                throw JinjaError.runtime("items filter requires an argument")
            }
            // Handle undefined values by returning empty array
            if value is UndefinedValue {
                return ArrayValue(value: [])
            }
            // Handle objects (mappings)
            if let objectValue = value as? ObjectValue {
                return ArrayValue(
                    value: objectValue.storage.map { (key, value) in
                        ArrayValue(value: [StringValue(value: key), value])
                    }
                )
            }

            throw JinjaError.runtime("Can only get item pairs from a mapping.")
        },
        "join": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("join filter requires an array")
            }
            let separator = (args.count > 1 && args[1] is StringValue) ? (args[1] as! StringValue).value : ""
            // Convert all values to strings before joining
            let stringValues = try arrayValue.value.map { value -> String in
                if let stringValue = value as? StringValue {
                    return stringValue.value
                } else {
                    // Convert other types to string using stringify function
                    return try stringify(value)
                }
            }
            return StringValue(value: stringValues.joined(separator: separator))
        },
        "last": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("last filter requires an array")
            }
            return arrayValue.value.last ?? UndefinedValue()
        },
        "length": { args, env in
            guard let arg = args.first else {
                throw JinjaError.runtime("length filter expects one argument")
            }

            if arg is UndefinedValue || arg is NullValue {
                return NumericValue(value: 0)
            }

            if let arrayValue = arg as? ArrayValue {
                return NumericValue(value: arrayValue.value.count)
            } else if let stringValue = arg as? StringValue {
                return NumericValue(value: stringValue.value.count)
            } else if let objectValue = arg as? ObjectValue {
                return NumericValue(value: objectValue.value.count)
            } else {
                throw JinjaError.runtime("Cannot get length of type: \(type(of: arg))")
            }
        },
        "list": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("list filter requires an array")
            }
            return arrayValue
        },
        "lower": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("lower filter requires a string")
            }
            return StringValue(value: stringValue.value.lowercased())
        },
        "map": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                // Handle None/empty case
                if args[0] is NullValue {
                    return ArrayValue(value: [])
                }
                throw JinjaError.runtime("map filter requires an array")
            }
            // Handle attribute mapping
            if args.count >= 2, let kwargs = args.last as? ObjectValue,
                let attribute = kwargs.value["attribute"] as? StringValue
            {
                let defaultValue = kwargs.value["default"]  // Get default value if provided
                return ArrayValue(
                    value: arrayValue.value.map { item in
                        if let objectValue = item as? ObjectValue {
                            if let value = objectValue.value[attribute.value] {
                                if value is UndefinedValue {
                                    // If value is explicitly undefined, return "None"
                                    return StringValue(value: "None")
                                }
                                if value is NullValue {
                                    // If value is explicitly null, return default if provided
                                    return defaultValue ?? StringValue(value: "None")
                                }
                                return value
                            } else {
                                // If attribute doesn't exist, use default
                                return defaultValue ?? StringValue(value: "None")
                            }
                        }
                        return defaultValue ?? StringValue(value: "None")
                    }
                )
            }
            // Handle function mapping by name
            if let functionName = args[1] as? StringValue {
                guard let filter = env.filters[functionName.value] else {
                    throw JinjaError.runtime("Unknown function: \(functionName.value)")
                }
                return ArrayValue(
                    value: try arrayValue.value.map { item in
                        try filter([item], env)
                    }
                )
            }
            throw JinjaError.runtime("map filter requires either an attribute name or a function name")
        },
        "min": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("min filter requires an array")
            }
            if arrayValue.value.isEmpty {
                return UndefinedValue()
            }
            if let numericValues = arrayValue.value as? [NumericValue] {
                let ints = numericValues.compactMap { $0.value as? Int }
                let doubles = numericValues.compactMap { $0.value as? Double }
                if !ints.isEmpty, doubles.isEmpty {
                    if let min = ints.min() {
                        return NumericValue(value: min)
                    } else {
                        throw JinjaError.runtime("min value of array in min filter could not be determined")
                    }
                } else if !doubles.isEmpty, ints.isEmpty {
                    if let min = doubles.min() {
                        return NumericValue(value: min)
                    } else {
                        throw JinjaError.runtime("min value of array in min filter could not be determined")
                    }
                } else {
                    throw JinjaError.runtime("min filter requires all array elements to be of type Int or Double")
                }
            } else if let stringValues = arrayValue.value as? [StringValue] {
                return StringValue(value: stringValues.map { $0.value }.min() ?? "")
            } else {
                throw JinjaError.runtime("min filter requires an array of numbers or strings")
            }
        },
        "max": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("max filter requires an array")
            }
            if arrayValue.value.isEmpty {
                return UndefinedValue()
            }
            if let numericValues = arrayValue.value as? [NumericValue] {
                let ints = numericValues.compactMap { $0.value as? Int }
                let doubles = numericValues.compactMap { $0.value as? Double }
                if !ints.isEmpty, doubles.isEmpty {
                    if let max = ints.max() {
                        return NumericValue(value: max)
                    } else {
                        throw JinjaError.runtime("max value of array in max filter cannot be determined")
                    }
                } else if !doubles.isEmpty, ints.isEmpty {
                    if let max = doubles.max() {
                        return NumericValue(value: max)
                    } else {
                        throw JinjaError.runtime("max value of array in max filter cannot be determined")
                    }
                } else {
                    throw JinjaError.runtime("max filter requires all array elements to be of type Int or Double")
                }
            } else if let stringValues = arrayValue.value as? [StringValue] {
                return StringValue(value: stringValues.map { $0.value }.max() ?? "")
            } else {
                throw JinjaError.runtime("max filter requires an array of numbers or strings")
            }
        },
        "pprint": { args, env in
            guard let value = args.first else {
                throw JinjaError.runtime("pprint filter expects one argument")
            }
            return StringValue(value: String(describing: value))
        },
        "random": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("random filter requires an array")
            }
            if let randomIndex = arrayValue.value.indices.randomElement() {
                return arrayValue.value[randomIndex]
            } else {
                return UndefinedValue()
            }
        },
        "reject": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("reject filter requires an array")
            }
            guard let testName = args[1] as? StringValue else {
                throw JinjaError.runtime("reject filter requires a test name")
            }
            guard let test = env.tests[testName.value] else {
                throw JinjaError.runtime("Unknown test '\(testName.value)'")
            }

            // Pre-compute additional arguments to avoid repeated array creation
            let additionalArgs = Array(args[2...])

            // Use compactMap for better functional style and performance
            let result = try arrayValue.value.compactMap { item -> (any RuntimeValue)? in
                let testArgs = [item] + additionalArgs
                return try !test(testArgs) ? item : nil
            }

            return ArrayValue(value: result)
        },
        "rejectattr": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("rejectattr filter requires an array")
            }
            guard let attribute = args[1] as? StringValue else {
                throw JinjaError.runtime("rejectattr filter requires an attribute name")
            }
            var result: [any RuntimeValue] = []
            for item in arrayValue.value {
                guard let objectValue = item as? ObjectValue,
                    let attrValue = objectValue.value[attribute.value]
                else {
                    continue
                }
                if args.count == 2 {
                    if !attrValue.bool() {
                        result.append(item)
                    }
                } else {
                    let testName = (args[2] as? StringValue)?.value ?? "defined"
                    guard let test = env.tests[testName] else {
                        throw JinjaError.runtime("Unknown test '\(testName)'")
                    }
                    // Correctly pass arguments to the test function
                    if try !test([attrValue]) {  // Note the negation (!) for rejectattr
                        result.append(item)
                    }
                }
            }
            return ArrayValue(value: result)
        },
        "replace": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("replace filter requires a string")
            }
            guard let oldValue = args[1] as? StringValue else {
                throw JinjaError.runtime("replace filter requires an old value string")
            }
            guard let newValue = args[2] as? StringValue else {
                throw JinjaError.runtime("replace filter requires a new value string")
            }
            let count = (args.count > 3 && args[3] is NumericValue) ? (args[3] as! NumericValue).value as! Int : Int.max
            return StringValue(
                value: stringValue.value.replacingOccurrences(
                    of: oldValue.value,
                    with: newValue.value,
                    options: [],
                    range: nil
                )
            )
        },
        "reverse": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("reverse filter requires an array")
            }
            return ArrayValue(value: arrayValue.value.reversed())
        },
        "round": { args, env in
            guard let value = args[0] as? NumericValue, let number = value.value as? Double else {
                throw JinjaError.runtime("round filter requires a number")
            }
            let precision = (args.count > 1 && args[1] is NumericValue) ? (args[1] as! NumericValue).value as! Int : 0
            let method = (args.count > 2 && args[2] is StringValue) ? (args[2] as! StringValue).value : "common"
            let factor = pow(10, Double(precision))
            let roundedNumber: Double
            if method == "common" {
                roundedNumber = round(number * factor) / factor
            } else if method == "ceil" {
                roundedNumber = ceil(number * factor) / factor
            } else if method == "floor" {
                roundedNumber = floor(number * factor) / factor
            } else {
                throw JinjaError.runtime("Invalid method for round filter")
            }
            return NumericValue(value: roundedNumber)
        },
        "safe": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("safe filter requires a string")
            }
            return stringValue  // In this minimal example, we don't handle marking strings as safe
        },
        "select": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("select filter requires an array")
            }
            guard let testName = args[1] as? StringValue else {
                throw JinjaError.runtime("select filter requires a test name")
            }
            guard let test = env.tests[testName.value] else {
                throw JinjaError.runtime("Unknown test '\(testName.value)'")
            }
            var result: [any RuntimeValue] = []
            for item in arrayValue.value {
                if try test([item]) {
                    result.append(item)
                }
            }
            return ArrayValue(value: result)
        },
        "selectattr": { args, env in
            guard let array = args[0] as? ArrayValue else {
                throw JinjaError.runtime("selectattr filter requires an array")
            }
            guard let attribute = args[1] as? StringValue else {
                throw JinjaError.runtime("selectattr filter requires an attribute name")
            }
            guard args.count > 2 else {
                throw JinjaError.runtime("selectattr filter requires a test")
            }
            var result: [any RuntimeValue] = []
            for item in array.value {
                if let obj = item as? ObjectValue,
                    let attrValue = obj.value[attribute.value]
                {
                    if args[2] is StringValue && args[2].bool() {
                        // Simple boolean test
                        if attrValue.bool() {
                            result.append(item)
                        }
                    } else if args.count > 3 {
                        // Test with comparison value
                        if let testName = (args[2] as? StringValue)?.value {
                            let testValue = args[3]
                            if testName == "equalto" {
                                // Handle equality test
                                if let strAttr = attrValue as? StringValue,
                                    let strTest = testValue as? StringValue
                                {
                                    if strAttr.value == strTest.value {
                                        result.append(item)
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return ArrayValue(value: result)
        },
        "slice": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("slice filter requires an array")
            }
            guard let slicesValue = args[1] as? NumericValue,
                let slices = slicesValue.value as? Int,
                slices > 0
            else {
                throw JinjaError.runtime("slice filter requires a positive number of slices")
            }

            let fillWith = args.count > 2 ? args[2] : nil
            let seq = arrayValue.value
            let length = seq.count
            let itemsPerSlice = length / slices
            let slicesWithExtra = length % slices
            var offset = 0

            var result: [[any RuntimeValue]] = []

            for sliceNumber in 0 ..< slices {
                let start = offset + sliceNumber * itemsPerSlice

                if sliceNumber < slicesWithExtra {
                    offset += 1
                }

                let end = offset + (sliceNumber + 1) * itemsPerSlice
                var tmp = Array(seq[start ..< end])

                if let fillWith = fillWith, sliceNumber >= slicesWithExtra {
                    tmp.append(fillWith)
                }

                result.append(tmp)
            }

            return ArrayValue(value: result.map { ArrayValue(value: $0) })
        },
        "sort": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("sort filter requires an array")
            }

            let reverse = args.count > 1 ? (args[1] as? BooleanValue)?.value ?? false : false
            let caseSensitive = args.count > 2 ? (args[2] as? BooleanValue)?.value ?? false : false
            let attributeStr = args.count > 3 ? (args[3] as? StringValue)?.value : nil

            // Helper function to get value from dot notation path
            func getValueFromPath(_ obj: any RuntimeValue, _ path: String) throws -> any RuntimeValue {
                let components = path.split(separator: ".")
                var current: any RuntimeValue = obj

                for component in components {
                    if let currentObj = current as? ObjectValue,
                        let value = currentObj.value[String(component)]
                    {
                        current = value
                    } else if let currentArray = current as? ArrayValue,
                        let index = Int(component),
                        index >= 0 && index < currentArray.value.count
                    {
                        current = currentArray.value[index]
                    } else {
                        throw JinjaError.runtime("Cannot access '\(component)' in path '\(path)'")
                    }
                }
                return current
            }

            // Helper function to compare RuntimeValues
            func compare(_ a: any RuntimeValue, _ b: any RuntimeValue) throws -> Bool {
                if let aStr = a as? StringValue, let bStr = b as? StringValue {
                    if caseSensitive {
                        return aStr.value < bStr.value
                    } else {
                        return aStr.value.lowercased() < bStr.value.lowercased()
                    }
                } else if let aNum = a as? NumericValue, let bNum = b as? NumericValue {
                    if let aInt = aNum.value as? Int, let bInt = bNum.value as? Int {
                        return aInt < bInt
                    } else if let aDouble = aNum.value as? Double, let bDouble = bNum.value as? Double {
                        return aDouble < bDouble
                    } else if let aInt = aNum.value as? Int, let bDouble = bNum.value as? Double {
                        return Double(aInt) < bDouble
                    } else if let aDouble = aNum.value as? Double, let bInt = bNum.value as? Int {
                        return aDouble < Double(bInt)
                    }
                }
                throw JinjaError.runtime("Cannot compare values of different types")
            }

            // Sort the array
            let sortedArray = try arrayValue.value.sorted { (a, b) -> Bool in
                if let attributeStr = attributeStr {
                    // Handle multiple attributes (comma-separated)
                    let attributes = attributeStr.split(separator: ",").map(String.init)

                    for attribute in attributes {
                        let aValue = try getValueFromPath(a, attribute.trimmingCharacters(in: .whitespaces))
                        let bValue = try getValueFromPath(b, attribute.trimmingCharacters(in: .whitespaces))

                        // If values are equal, continue to next attribute
                        if try compare(aValue, bValue) == compare(bValue, aValue) {
                            continue
                        }

                        return reverse ? try !compare(aValue, bValue) : try compare(aValue, bValue)
                    }
                    // All attributes were equal
                    return false
                } else {
                    return reverse ? try !compare(a, b) : try compare(a, b)
                }
            }

            return ArrayValue(value: sortedArray)
        },
        "string": { args, env in
            guard let arg = args.first else {
                throw JinjaError.runtime("string filter expects one argument")
            }
            // In Jinja2 in Python, the `string` filter calls Python's `str` function on dicts, which which uses single quotes for strings. Here we're using double quotes in `tojson`, which is probably better for LLMs anyway, but this will result in differences with output from Jinja2.
            return try StringValue(value: stringify(arg, whitespaceControl: true))
        },
        "striptags": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("striptags filter requires a string")
            }
            // A very basic implementation to remove HTML tags
            let tagPattern = #"<[^>]+>"#
            let noTagsString = stringValue.value.replacingOccurrences(
                of: tagPattern,
                with: "",
                options: .regularExpression
            )
            return StringValue(value: noTagsString)
        },
        "sum": { args, env in
            guard let arrayValue = args[0] as? ArrayValue else {
                throw JinjaError.runtime("sum filter requires an array")
            }

            // Get attribute and start value from arguments
            let attribute = args.count > 1 ? args[1] : nil
            let start: Double = {
                if args.count > 2, let numericValue = args[2] as? NumericValue {
                    if let intValue = numericValue.value as? Int {
                        return Double(intValue)
                    } else if let doubleValue = numericValue.value as? Double {
                        return doubleValue
                    }
                }
                return 0.0
            }()

            // Helper function to get value based on attribute
            func getValue(_ item: any RuntimeValue) throws -> Double {
                if let attribute = attribute {
                    // Handle string attribute (object property)
                    if let strAttr = attribute as? StringValue,
                        let objectValue = item as? ObjectValue,
                        let attrValue = objectValue.value[strAttr.value]
                    {
                        if let numericValue = attrValue as? NumericValue {
                            if let intValue = numericValue.value as? Int {
                                return Double(intValue)
                            } else if let doubleValue = numericValue.value as? Double {
                                return doubleValue
                            }
                        }
                        throw JinjaError.runtime("Attribute '\(strAttr.value)' is not numeric")
                    }
                    // Handle integer attribute (array/string index)
                    else if let numAttr = attribute as? NumericValue,
                        let index = numAttr.value as? Int
                    {
                        if let arrayValue = item as? ArrayValue {
                            guard index >= 0 && index < arrayValue.value.count else {
                                throw JinjaError.runtime("Index \(index) out of range")
                            }
                            if let numericValue = arrayValue.value[index] as? NumericValue {
                                if let intValue = numericValue.value as? Int {
                                    return Double(intValue)
                                } else if let doubleValue = numericValue.value as? Double {
                                    return doubleValue
                                }
                            }
                            throw JinjaError.runtime("Value at index \(index) is not numeric")
                        }
                    }
                    throw JinjaError.runtime("Cannot get attribute '\(try stringify(attribute))' from item")
                } else {
                    // No attribute - use item directly
                    if let numericValue = item as? NumericValue {
                        if let intValue = numericValue.value as? Int {
                            return Double(intValue)
                        } else if let doubleValue = numericValue.value as? Double {
                            return doubleValue
                        }
                    }
                    throw JinjaError.runtime("Item is not numeric")
                }
            }

            // Sum all values
            var result = start
            for item in arrayValue.value {
                do {
                    result += try getValue(item)
                } catch {
                    throw JinjaError.runtime("Could not sum items: \(error.localizedDescription)")
                }
            }

            // Return result as NumericValue
            // If the result has no decimal part, return as Int
            if result.truncatingRemainder(dividingBy: 1) == 0 {
                return NumericValue(value: Int(result))
            }
            return NumericValue(value: result)
        },
        "title": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("title filter requires a string")
            }

            // Split the string by spaces, hyphens, and opening brackets/braces/parentheses
            let pattern = "([-\\s(\\{\\[<]+)"
            let regex = try! NSRegularExpression(pattern: pattern, options: [])
            let str = stringValue.value
            let range = NSRange(str.startIndex ..< str.endIndex, in: str)

            // Split the string and keep the delimiters
            let matches = regex.matches(in: str, options: [], range: range)
            var parts: [String] = []
            var currentIndex = str.startIndex

            // Add the first part if it exists
            if let firstMatch = matches.first,
                let firstMatchRange = Range(firstMatch.range, in: str)
            {
                if currentIndex < firstMatchRange.lowerBound {
                    parts.append(String(str[currentIndex ..< firstMatchRange.lowerBound]))
                }
                parts.append(String(str[firstMatchRange]))
                currentIndex = firstMatchRange.upperBound
            }

            // Add remaining parts and delimiters
            for i in 1 ..< matches.count {
                if let matchRange = Range(matches[i].range, in: str) {
                    if currentIndex < matchRange.lowerBound {
                        parts.append(String(str[currentIndex ..< matchRange.lowerBound]))
                    }
                    parts.append(String(str[matchRange]))
                    currentIndex = matchRange.upperBound
                }
            }

            // Add the last part if it exists
            if currentIndex < str.endIndex {
                parts.append(String(str[currentIndex ..< str.endIndex]))
            }

            // Process each part and join them
            let result = parts.filter { !$0.isEmpty }.map { part -> String in
                if part.matches(of: try! Regex(pattern)).isEmpty {
                    // This is a word part, not a delimiter
                    if let first = part.first {
                        return String(first).uppercased() + part.dropFirst().lowercased()
                    }
                    return part
                }
                // This is a delimiter, keep it as is
                return part
            }.joined()

            return StringValue(value: result)
        },
        "trim": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("trim filter requires a string")
            }
            return StringValue(value: stringValue.value.trimmingCharacters(in: .whitespacesAndNewlines))
        },
        "truncate": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("truncate filter requires a string")
            }
            let length = (args.count > 1 && args[1] is NumericValue) ? (args[1] as! NumericValue).value as! Int : 255
            let killwords = (args.count > 2 && args[2] is BooleanValue) ? (args[2] as! BooleanValue).value : false
            let end = (args.count > 3 && args[3] is StringValue) ? (args[3] as! StringValue).value : "..."
            if stringValue.value.count <= length {
                return stringValue
            }
            if killwords {
                return StringValue(value: String(stringValue.value.prefix(length - end.count)) + end)
            } else {
                let truncated = String(stringValue.value.prefix(length - end.count))
                if let lastSpace = truncated.lastIndex(of: " ") {
                    return StringValue(value: String(truncated[..<lastSpace]) + end)
                } else {
                    return StringValue(value: truncated + end)
                }
            }
        },
        "unique": { args, env in
            // Handle different iterable types
            func getIterableItems(_ value: any RuntimeValue) throws -> [any RuntimeValue] {
                switch value {
                case let arrayValue as ArrayValue:
                    return arrayValue.value
                case let stringValue as StringValue:
                    // Always split string into characters as StringValues
                    return stringValue.value.map { StringValue(value: String($0)) }
                case let objectValue as ObjectValue:
                    return objectValue.storage.map { key, value in
                        ArrayValue(value: [StringValue(value: key), value])
                    }
                default:
                    throw JinjaError.runtime("Value must be iterable (array, string, or object)")
                }
            }
            // Get the input iterable
            guard let input = args.first else {
                throw JinjaError.runtime("unique filter requires an iterable")
            }
            let caseSensitive = args.count > 1 ? (args[1] as? BooleanValue)?.value ?? false : false
            let attribute = args.count > 2 ? args[2] : nil
            // Helper function to get value based on attribute
            func getValue(_ item: any RuntimeValue) throws -> String {
                if let attribute = attribute {
                    // Handle string attribute (object property)
                    if let strAttr = attribute as? StringValue,
                        let objectValue = item as? ObjectValue
                    {
                        // Support dot notation
                        let components = strAttr.value.split(separator: ".")
                        var current: any RuntimeValue = objectValue

                        for component in components {
                            if let currentObj = current as? ObjectValue,
                                let value = currentObj.value[String(component)]
                            {
                                current = value
                            } else {
                                throw JinjaError.runtime("Cannot access '\(component)' in path '\(strAttr.value)'")
                            }
                        }
                        return try stringify(current)
                    }
                    // Handle integer attribute (array/string index)
                    else if let numAttr = attribute as? NumericValue,
                        let index = numAttr.value as? Int
                    {
                        if let stringValue = item as? StringValue {
                            let str = stringValue.value
                            guard index >= 0 && index < str.count else {
                                throw JinjaError.runtime("Index \(index) out of range")
                            }
                            let stringIndex = str.index(str.startIndex, offsetBy: index)
                            return String(str[stringIndex])
                        } else if let arrayValue = item as? ArrayValue {
                            guard index >= 0 && index < arrayValue.value.count else {
                                throw JinjaError.runtime("Index \(index) out of range")
                            }
                            return try stringify(arrayValue.value[index])
                        }
                    }
                }
                // No attribute - use item directly
                return try stringify(item)
            }
            var seen: [String: Bool] = [:]
            var result: [any RuntimeValue] = []
            // Process all items from the iterable
            let items = try getIterableItems(input)
            for item in items {
                let key = try getValue(item)
                let lookupKey = caseSensitive ? key : key.lowercased()

                if seen[lookupKey] == nil {
                    seen[lookupKey] = true
                    result.append(item)
                }
            }
            return ArrayValue(value: result)
        },
        "upper": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("upper filter requires a string")
            }
            return StringValue(value: stringValue.value.uppercased())
        },
        "urlencode": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("urlencode filter requires a string")
            }

            let encodedString = stringValue.value.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
            return StringValue(value: encodedString)
        },
        "urlize": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("urlize filter requires a string")
            }
            let trimUrlLimit =
                (args.count > 1 && args[1] is NumericValue) ? (args[1] as! NumericValue).value as? Int : nil
            let nofollow = (args.count > 2 && args[2] is BooleanValue) ? (args[2] as! BooleanValue).value : false
            let target = (args.count > 3 && args[3] is StringValue) ? (args[3] as! StringValue).value : nil
            let urlPattern =
                #"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"#
            var urlizedString = stringValue.value
            if let regex = try? NSRegularExpression(pattern: urlPattern, options: []) {
                let nsRange = NSRange(
                    stringValue.value.startIndex ..< stringValue.value.endIndex,
                    in: stringValue.value
                )
                let matches = regex.matches(in: stringValue.value, options: [], range: nsRange)

                for match in matches.reversed() {
                    let urlRange = Range(match.range, in: stringValue.value)!
                    let url = String(stringValue.value[urlRange])
                    var trimmedUrl = url
                    if let limit = trimUrlLimit, url.count > limit {
                        trimmedUrl = String(url.prefix(limit)) + "..."
                    }
                    var link = "<a href=\"\(url)\""
                    if nofollow {
                        link += " rel=\"nofollow\""
                    }
                    if let target = target {
                        link += " target=\"\(target)\""
                    }
                    link += ">\(trimmedUrl)</a>"
                    urlizedString.replaceSubrange(urlRange, with: link)
                }
            }

            return StringValue(value: urlizedString)
        },
        "wordcount": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("wordcount filter requires a string")
            }
            let words = stringValue.value.split(separator: " ")
            return NumericValue(value: words.count)
        },
        "wordwrap": { args, env in
            guard let stringValue = args[0] as? StringValue else {
                throw JinjaError.runtime("wordwrap filter requires a string")
            }
            let width = (args.count > 1 && args[1] is NumericValue) ? (args[1] as! NumericValue).value as! Int : 79
            let breakLongWords = (args.count > 2 && args[2] is BooleanValue) ? (args[2] as! BooleanValue).value : true
            let wrapString = (args.count > 3 && args[3] is StringValue) ? (args[3] as! StringValue).value : "\n"
            var result = ""
            var currentLineLength = 0
            for word in stringValue.value.split(separator: " ", omittingEmptySubsequences: false) {
                if currentLineLength + word.count > width {
                    if currentLineLength > 0 {
                        result += wrapString
                        currentLineLength = 0
                    }
                    if word.count > width && breakLongWords {
                        var remainingWord = word[...]
                        while remainingWord.count > width {
                            result += remainingWord.prefix(width)
                            result += wrapString
                            remainingWord = remainingWord.dropFirst(width)
                        }
                        if !remainingWord.isEmpty {
                            result += remainingWord
                            currentLineLength = remainingWord.count
                        }
                        continue
                    }
                }
                if !result.isEmpty && currentLineLength == 0 {
                    result += word
                    currentLineLength = word.count
                } else {
                    if !result.isEmpty {
                        result += " "
                        currentLineLength += 1
                    }
                    result += word
                    currentLineLength += word.count
                }
            }
            return StringValue(value: result)
        },
        "xmlattr": { args, env in
            guard let dict = args[0] as? ObjectValue else {
                throw JinjaError.runtime("xmlattr filter requires a dictionary")
            }
            let autospace = args.count > 1 ? (args[1] as? BooleanValue)?.value ?? true : true
            var result = ""
            for (key, value) in dict.storage {
                if !(value is UndefinedValue) && !(value is NullValue) {
                    if autospace {
                        result += " "
                    }
                    if let stringValue = value as? StringValue {
                        result +=
                            "\(key)=\"\(stringValue.value.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "\"", with: "&quot;"))\""
                    } else {
                        result += "\(key)=\"\(value)\""
                    }
                }
            }
            return StringValue(value: result)
        },
        "tojson": { args, env in
            guard let firstArg = args.first else {
                throw JinjaError.runtime("tojson filter expects at least one argument")
            }
            var indent: Int? = nil
            if args.count > 1, let kwargs = args.last as? ObjectValue,
                let indentArg = kwargs.value["indent"] as? NumericValue,
                let indentInt = indentArg.value as? Int
            {
                indent = indentInt
            }
            return try StringValue(value: toJSON(firstArg, indent: indent, whitespaceControl: false))
        },
    ]

    init(parent: Environment? = nil) {
        self.parent = parent
    }

    //    func isFunction<T>(_ value: Any, functionType: T.Type) -> Bool {
    //        return value is T
    //    }

    func convertToRuntimeValues(input: Any?) throws -> any RuntimeValue {
        // Handle already converted RuntimeValue
        if let runtimeValue = input as? any RuntimeValue {
            return runtimeValue
        }
        // Handle nil values
        if input == nil {
            return NullValue()
        }
        if case Optional<Any>.none = input {
            return NullValue()
        }
        // Helper function to handle any OrderedDictionary type
        func convertOrderedDictionary<T>(_ dict: OrderedDictionary<String, T>) throws -> ObjectValue {
            var object: [String: any RuntimeValue] = [:]
            var keyOrder: [String] = []

            for (key, value) in dict {
                // Crucial: Convert Optional<T> to T, using NullValue if nil
                let convertedValue = (value as Any?) ?? NullValue()
                object[key] = try self.convertToRuntimeValues(input: convertedValue)
                keyOrder.append(key)
            }
            return ObjectValue(value: object, keyOrder: keyOrder)
        }
        // Handle other values
        switch input {
        case let value as Bool:
            return BooleanValue(value: value)
        case let value as Int:
            return NumericValue(value: value)
        case let value as Double:
            return NumericValue(value: value)
        case let value as Float:
            return NumericValue(value: value)
        case let value as String:
            return StringValue(value: value)
        case let data as Data:
            guard let string = String(data: data, encoding: .utf8) else {
                throw JinjaError.runtime("Failed to convert data to string")
            }
            return StringValue(value: string)
        case let fn as (String) throws -> Void:
            return FunctionValue { args, _ in
                guard let stringArg = args[0] as? StringValue else {
                    throw JinjaError.runtime("Argument must be a StringValue")
                }
                try fn(stringArg.value)
                return NullValue()
            }
        case let fn as (Bool) throws -> Void:
            return FunctionValue { args, _ in
                guard let boolArg = args[0] as? BooleanValue else {
                    throw JinjaError.runtime("Argument must be a BooleanValue")
                }
                try fn(boolArg.value)
                return NullValue()
            }
        case let fn as (Int, Int?, Int) -> [Int]:
            return FunctionValue { args, _ in
                guard args.count > 0, let arg0 = args[0] as? NumericValue, let int0 = arg0.value as? Int else {
                    throw JinjaError.runtime("First argument must be an Int")
                }
                var int1: Int? = nil
                if args.count > 1 {
                    if let numericValue = args[1] as? NumericValue, let tempInt1 = numericValue.value as? Int {
                        int1 = tempInt1
                    } else if !(args[1] is NullValue) {  // Accept NullValue for optional second argument
                        throw JinjaError.runtime("Second argument must be an Int or nil")
                    }
                }
                var int2: Int = 1
                if args.count > 2 {
                    if let numericValue = args[2] as? NumericValue, let tempInt2 = numericValue.value as? Int {
                        int2 = tempInt2
                    } else {
                        throw JinjaError.runtime("Third argument must be an Int")
                    }
                }
                let result = fn(int0, int1, int2)
                return ArrayValue(value: result.map { NumericValue(value: $0) })
            }
        case let values as [Any?]:
            let items = try values.map { try self.convertToRuntimeValues(input: $0) }
            return ArrayValue(value: items)
        case let orderedDict as OrderedDictionary<String, String>:
            return try convertOrderedDictionary(orderedDict)
        case let orderedDict as OrderedDictionary<String, OrderedDictionary<String, Any>>:
            return try convertOrderedDictionary(orderedDict)
        case let orderedDict as OrderedDictionary<String, OrderedDictionary<String, String>>:
            return try convertOrderedDictionary(orderedDict)
        case let orderedDict as OrderedDictionary<String, Any?>:
            return try convertOrderedDictionary(orderedDict)
        case let orderedDict as OrderedDictionary<String, Any>:
            return try convertOrderedDictionary(orderedDict)
        case let dictionary as [String: Any?]:
            var object: [String: any RuntimeValue] = [:]
            var keyOrder: [String] = []
            for (key, value) in dictionary {
                object[key] = try self.convertToRuntimeValues(input: value)
                keyOrder.append(key)
            }
            return ObjectValue(value: object, keyOrder: keyOrder)
        default:
            throw JinjaError.runtime(
                "Cannot convert to runtime value: \(String(describing: input)) type:\(type(of: input))"
            )
        }
    }

    @discardableResult
    func set(name: String, value: Any) throws -> any RuntimeValue {
        let runtimeValue = try self.convertToRuntimeValues(input: value)
        return try self.declareVariable(name: name, value: runtimeValue)
    }

    private func declareVariable(name: String, value: any RuntimeValue) throws -> any RuntimeValue {
        if self.variables.keys.contains(name) {
            throw JinjaError.syntax("Variable already declared: \(name)")
        }

        self.variables[name] = value
        return value
    }

    @discardableResult
    func setVariable(name: String, value: any RuntimeValue) throws -> any RuntimeValue {
        self.variables[name] = value
        return value
    }

    private func resolve(name: String) throws -> Environment {
        if self.variables.keys.contains(name) {
            return self
        }

        if let parent = self.parent {
            return try parent.resolve(name: name)
        }

        throw JinjaError.runtime("Unknown variable: \(name)")
    }

    func lookupVariable(name: String) -> any RuntimeValue {
        do {
            // Look up the variable in the environment chain
            let env = try self.resolve(name: name)

            // Get the value, handling potential conversions from Swift native types
            if let value = env.variables[name] {
                // If we have a raw Swift boolean, ensure it's properly converted to BooleanValue
                if let boolValue = value.value as? Bool {
                    return BooleanValue(value: boolValue)
                }
                return value
            }

            // Variable doesn't exist
            return UndefinedValue()
        } catch {
            // Cannot resolve variable name
            return UndefinedValue()
        }
    }

    // Filters

    private func doDefault(_ args: [any RuntimeValue], _ env: Environment) throws -> any RuntimeValue {
        let value = args[0]
        let defaultValue = args.count > 1 ? args[1] : StringValue(value: "")
        let boolean = args.count > 2 ? (args[2] as? BooleanValue)?.value ?? false : false

        if value is UndefinedValue {
            return defaultValue
        }

        if boolean {
            if !value.bool() {
                return defaultValue
            }
            // If it's a boolean value, return its string representation
            if let boolValue = value as? BooleanValue {
                return StringValue(value: String(boolValue.value))
            }
        }

        return value
    }

    private func doEscape(_ args: [any RuntimeValue], _ env: Environment) throws -> any RuntimeValue {
        guard let stringValue = args[0] as? StringValue else {
            throw JinjaError.runtime("escape filter requires a string")
        }
        return StringValue(
            value: stringValue.value.replacingOccurrences(of: "&", with: "&amp;")
                .replacingOccurrences(of: "<", with: "&lt;")
                .replacingOccurrences(of: ">", with: "&gt;")
                .replacingOccurrences(of: "\"", with: "&quot;")
                .replacingOccurrences(of: "'", with: "&#39;")
        )
    }

    private func doEqualTo(_ args: [any RuntimeValue]) -> Bool {
        if args.count == 2 {
            if let left = args[0] as? StringValue, let right = args[1] as? StringValue {
                return left.value == right.value
            } else if let left = args[0] as? NumericValue, let right = args[1] as? NumericValue,
                let leftInt = left.value as? Int, let rightInt = right.value as? Int
            {
                return leftInt == rightInt
            } else if let left = args[0] as? BooleanValue, let right = args[1] as? BooleanValue {
                return left.value == right.value
            } else {
                return false
            }
        } else {
            return false
        }
    }

    // Tests

    private func doGreaterThan(_ args: [any RuntimeValue]) throws -> Bool {
        if let left = args[0] as? StringValue, let right = args[1] as? StringValue {
            return left.value > right.value
        } else if let left = args[0] as? NumericValue, let right = args[1] as? NumericValue {
            if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                return leftInt > rightInt
            } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                return leftDouble > rightDouble
            } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                return Double(leftInt) > rightDouble
            } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                return leftDouble > Double(rightInt)
            }
        }
        throw JinjaError.runtime("Cannot compare values of different types")
    }

    private func doGreaterThanOrEqual(_ args: [any RuntimeValue]) throws -> Bool {
        return try doGreaterThan(args) || doEqualTo(args)
    }

    private func doLessThan(_ args: [any RuntimeValue]) throws -> Bool {
        if let left = args[0] as? StringValue, let right = args[1] as? StringValue {
            return left.value < right.value
        } else if let left = args[0] as? NumericValue, let right = args[1] as? NumericValue {
            if let leftInt = left.value as? Int, let rightInt = right.value as? Int {
                return leftInt < rightInt
            } else if let leftDouble = left.value as? Double, let rightDouble = right.value as? Double {
                return leftDouble < rightDouble
            } else if let leftInt = left.value as? Int, let rightDouble = right.value as? Double {
                return Double(leftInt) < rightDouble
            } else if let leftDouble = left.value as? Double, let rightInt = right.value as? Int {
                return leftDouble < Double(rightInt)
            }
        }
        throw JinjaError.runtime("Cannot compare values of different types")
    }

    private func doLessThanOrEqual(_ args: [any RuntimeValue]) throws -> Bool {
        return try doLessThan(args) || doEqualTo(args)
    }

    /// Formats a date using strftime-style format specifiers
    /// - Parameters:
    ///   - date: The date to format
    ///   - format: A strftime-compatible format string
    /// - Returns: The formatted date string
    static func formatDate(_ date: Date, withFormat format: String) -> String {

        let calendar = Calendar.current
        let components = calendar.dateComponents(
            [
                .year, .month, .day, .weekday, .hour, .minute, .second, .nanosecond, .timeZone, .weekOfYear,
                .yearForWeekOfYear, .weekdayOrdinal, .quarter,
            ],
            from: date
        )

        var result = ""
        var i = 0

        while i < format.count {
            let currentIndex = format.index(format.startIndex, offsetBy: i)
            let currentChar = format[currentIndex]

            if currentChar == "%" && i + 1 < format.count {
                let nextIndex = format.index(format.startIndex, offsetBy: i + 1)
                let nextChar = format[nextIndex]

                // Check for non-padded variant
                var isPadded = true
                var formatChar = nextChar

                if nextChar == "-" && i + 2 < format.count {
                    isPadded = false
                    let formatCharIndex = format.index(format.startIndex, offsetBy: i + 2)
                    formatChar = format[formatCharIndex]
                    i += 1  // Skip the "-" character
                }

                switch formatChar {
                case "a":
                    let formatter = DateFormatter()
                    formatter.dateFormat = "EEE"
                    result += formatter.string(from: date)
                case "A":
                    let formatter = DateFormatter()
                    formatter.dateFormat = "EEEE"
                    result += formatter.string(from: date)
                case "w":
                    let weekday = (components.weekday ?? 1) - 1
                    result += "\(weekday)"
                case "d":
                    let day = components.day ?? 1
                    result += isPadded ? String(format: "%02d", day) : "\(day)"
                case "b":
                    let formatter = DateFormatter()
                    formatter.dateFormat = "MMM"
                    result += formatter.string(from: date)
                case "B":
                    let formatter = DateFormatter()
                    formatter.dateFormat = "MMMM"
                    result += formatter.string(from: date)
                case "m":
                    let month = components.month ?? 1
                    result += isPadded ? String(format: "%02d", month) : "\(month)"
                case "y":
                    let year = components.year ?? 0
                    let shortYear = year % 100
                    result += isPadded ? String(format: "%02d", shortYear) : "\(shortYear)"
                case "Y":
                    let year = components.year ?? 0
                    result += "\(year)"
                case "H":
                    let hour = components.hour ?? 0
                    result += isPadded ? String(format: "%02d", hour) : "\(hour)"
                case "I":
                    var hour12 = (components.hour ?? 0) % 12
                    if hour12 == 0 { hour12 = 12 }
                    result += isPadded ? String(format: "%02d", hour12) : "\(hour12)"
                case "p":
                    let hour = components.hour ?? 0
                    result += hour < 12 ? "AM" : "PM"
                case "M":
                    let minute = components.minute ?? 0
                    result += isPadded ? String(format: "%02d", minute) : "\(minute)"
                case "S":
                    let second = components.second ?? 0
                    result += isPadded ? String(format: "%02d", second) : "\(second)"
                case "f":
                    let nano = components.nanosecond ?? 0
                    let micro = nano / 1000
                    result += String(format: "%06d", micro)
                case "z":
                    guard let timeZone = components.timeZone else {
                        result += "+0000"
                        break
                    }
                    let hours = timeZone.secondsFromGMT() / 3600
                    let minutes = abs(timeZone.secondsFromGMT() % 3600) / 60
                    let sign = hours >= 0 ? "+" : "-"
                    result += "\(sign)\(String(format: "%02d", abs(hours)))\(String(format: "%02d", minutes))"
                case "Z":
                    guard let timeZone = components.timeZone else {
                        result += ""
                        break
                    }
                    result += timeZone.abbreviation() ?? ""
                case "j":
                    let dayOfYear = calendar.ordinality(of: .day, in: .year, for: date) ?? 1
                    result += isPadded ? String(format: "%03d", dayOfYear) : "\(dayOfYear)"
                case "U":
                    var cal = Calendar(identifier: .gregorian)
                    cal.firstWeekday = 1  // Sunday
                    let week = cal.component(.weekOfYear, from: date)
                    result += String(format: "%02d", week)
                case "W":
                    var cal = Calendar(identifier: .gregorian)
                    cal.firstWeekday = 2  // Monday
                    let week = cal.component(.weekOfYear, from: date)
                    result += String(format: "%02d", week)
                case "c":
                    let formatter = DateFormatter()
                    formatter.dateStyle = .full
                    formatter.timeStyle = .full
                    result += formatter.string(from: date)
                case "x":
                    let formatter = DateFormatter()
                    formatter.dateStyle = .short
                    formatter.timeStyle = .none
                    result += formatter.string(from: date)
                case "X":
                    let formatter = DateFormatter()
                    formatter.dateStyle = .none
                    formatter.timeStyle = .medium
                    result += formatter.string(from: date)
                case "%":
                    result += "%"
                default:
                    // Unknown format, just append as is
                    result += "%\(formatChar)"
                }

                i += 2  // Skip the % and the format character
            } else {
                result.append(currentChar)
                i += 1
            }
        }
        return result
    }
}
