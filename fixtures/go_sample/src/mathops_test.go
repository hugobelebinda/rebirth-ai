package mathops

import "testing"

func TestAdd(t *testing.T) {
	if Add(2, 3) != 5 {
		t.Fatalf("add")
	}
}

func TestMultiply(t *testing.T) {
	if Multiply(4, 5) != 20 {
		t.Fatalf("multiply")
	}
}

func TestSummarize(t *testing.T) {
	if Summarize(1, 2) != "sum=3" {
		t.Fatalf("summarize")
	}
}
