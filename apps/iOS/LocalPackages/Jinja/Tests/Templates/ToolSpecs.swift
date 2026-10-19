//
//  ToolSpecs.swift
//  Jinja
//
//  Created by Anthony DePasquale on 02.01.2025.
//

import OrderedCollections

struct ToolSpec {
    static let getCurrentWeather = OrderedDictionary(uniqueKeysWithValues: [
        ("type", "function") as (String, Any),
        (
            "function",
            OrderedDictionary(uniqueKeysWithValues: [
                ("name", "get_current_weather") as (String, Any),
                ("description", "Get the current weather in a given location") as (String, Any),
                (
                    "parameters",
                    OrderedDictionary(uniqueKeysWithValues: [
                        ("type", "object") as (String, Any),
                        (
                            "properties",
                            OrderedDictionary(uniqueKeysWithValues: [
                                (
                                    "location",
                                    OrderedDictionary(uniqueKeysWithValues: [
                                        ("type", "string") as (String, Any),
                                        ("description", "The city and state, e.g. San Francisco, CA")
                                            as (String, Any),
                                    ])
                                ) as (String, Any),
                                (
                                    "unit",
                                    OrderedDictionary(uniqueKeysWithValues: [
                                        ("type", "string") as (String, Any),
                                        ("enum", ["celsius", "fahrenheit"]) as (String, Any),
                                    ])
                                ) as (String, Any),
                            ])
                        ) as (String, Any),
                        ("required", ["location"]) as (String, Any),
                    ])
                ) as (String, Any),
            ])
        ) as (String, Any),
    ])
}
