// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MathOps",
    products: [
        .library(name: "MathOps", targets: ["MathOps"]),
    ],
    targets: [
        .target(name: "MathOps"),
        .testTarget(name: "MathOpsTests", dependencies: ["MathOps"]),
    ]
)
