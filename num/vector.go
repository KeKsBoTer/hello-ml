package num

import "math"

type Vector []float64

func NewVector(length int) *Vector {
	m := make(Vector, length)
	return &m
}

func (v Vector) Length() float64 {
	qSum := 0.0
	for _, p := range v {
		qSum += p * p
	}
	return math.Sqrt(qSum)
}

func (v Vector) Normalize() *Vector {
	l := v.Length()
	cp := *NewVector(len(v))
	for i, p := range v {
		cp[i] = p * l
	}
	return &cp
}

func (v Vector) Dot(v2 Vector) float64 {
	sum := 0.0
	for i, p := range v {
		sum += p * v2[i]
	}
	return sum
}

func (v Vector) Add(v2 Vector) *Vector {
	r := *NewVector(len(v))
	for i, p := range v {
		r[i] = p + v2[i]
	}
	return &r
}

func (v Vector) Apply(fn func(float64) float64) *Vector {
	r := *NewVector(len(v))
	for i, p := range v {
		r[i] = fn(p)
	}
	return &r
}
