public enum MathOps {
    public static func add(_ a: Int, _ b: Int) -> Int { a + b }

    public static func multiply(_ a: Int, _ b: Int) -> Int {
        // BUG: should multiply
        return a + b
    }

    public static func summarize(_ a: Int, _ b: Int) -> String {
        "sum=\(add(a, b))"
    }
}
