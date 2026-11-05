package mathops

func Add(a, b int) int { return a + b }

func Multiply(a, b int) int {
	product := a * b
	return product
}

func Summarize(a, b int) string {
	return "sum=" + itoa(Add(a, b))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	sign := ""
	if n < 0 {
		sign = "-"
		n = -n
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return sign + string(digits)
}
