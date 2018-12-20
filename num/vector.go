package num

import "math"

type Vector []float64

// NewVector creates new vector with given length
// e.g NewVector(3) => [0,0,0]
func NewVector(length int) *Vector {
	m := make(Vector, length)
	return &m
}

// Length returns the length of a vector
// Lenght([a,b,c]) = sqrt(a^2 + b^2 + c^2)
func (v Vector) Length() float64 {
	qSum := 0.0
	for _, p := range v {
		qSum += p * p
	}
	return math.Sqrt(qSum)
}

// Normalize scales a vector so that its length is equal to one
func (v Vector) Normalize() *Vector {
	l := v.Length()
	cp := *NewVector(len(v))
	for i, p := range v {
		cp[i] = p * l
	}
	return &cp
}

// Dot calculates the dot product with another vector
// e.g. [x1,y1].Dot([x2,y2]) = x1*x2 + y1*y2
func (v Vector) Dot(v2 Vector) float64 {
	sum := 0.0
	for i, p := range v {
		sum += p * v2[i]
	}
	return sum
}

// Add adds another array value by value
func (v Vector) Add(v2 Vector) *Vector {
	r := *NewVector(len(v))
	for i, p := range v {
		r[i] = p + v2[i]
	}
	return &r
}

// Sub substracts a array value by value
func (v Vector) Sub(v2 Vector) *Vector {
	r := *NewVector(len(v))
	for i, p := range v {
		r[i] = p - v2[i]
	}
	return &r
}

// Apply a function to every value of the vector
func (v Vector) Apply(fn func(float64) float64) *Vector {
	r := *NewVector(len(v))
	for i, p := range v {
		r[i] = fn(p)
	}
	return &r
}

// Cost of a vector to an expected vector
func (v Vector) Cost(expected Vector) float64 {
	sum := 0.0
	for i, p := range v {
		diff := p - expected[i]
		sum += diff * diff
	}
	return sum
}

// Max returns index with highes value
func (v Vector) Max() int {
	maxIndex, maxValue := -1, -math.MaxFloat64
	for i, p := range v {
		if p > maxValue {
			maxIndex, maxValue = i, p
		}
	}
	return maxIndex
}
