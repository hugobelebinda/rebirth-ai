pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(a: i32, b: i32) -> i32 {
    let product = a * b;
    product
}

pub fn summarize(a: i32, b: i32) -> String {
    format!("sum={}", add(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(4, 5), 20);
    }

    #[test]
    fn test_summarize() {
        assert_eq!(summarize(1, 2), "sum=3");
    }
}
