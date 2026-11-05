import XCTest
@testable import MathOps

final class MathOpsTests: XCTestCase {
    func testAdd() {
        XCTAssertEqual(MathOps.add(2, 3), 5)
    }

    func testMultiply() {
        XCTAssertEqual(MathOps.multiply(4, 5), 20)
    }

    func testSummarize() {
        XCTAssertEqual(MathOps.summarize(1, 2), "sum=3")
    }
}
