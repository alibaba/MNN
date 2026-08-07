// swift-tools-version: 5.9

import PackageDescription

let package = Package(
	name: "ActivityIndicatorView",
	platforms: [
		.macOS(.v10_15),
        .iOS(.v13),
        .watchOS(.v6),
        .tvOS(.v13)
    ],
    products: [
    	.library(
    		name: "ActivityIndicatorView", 
    		targets: ["ActivityIndicatorView"]
    	)
    ],
    targets: [
    	.target(
    		name: "ActivityIndicatorView",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        )
    ]
)
