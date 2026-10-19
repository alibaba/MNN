// swift-tools-version: 5.9

import PackageDescription

let package = Package(
	name: "FloatingButton",
	platforms: [
        .iOS(.v14),
		.macOS(.v11),
        .tvOS(.v14),
        .watchOS(.v7)
    ],
    products: [
    	.library(
    		name: "FloatingButton", 
    		targets: ["FloatingButton"]
    	)
    ],
    targets: [
    	.target(
    		name: "FloatingButton",
            dependencies: [],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        )
    ]
)
