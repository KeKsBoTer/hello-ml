package main

import (
	"math/rand"

	"github.com/KeKsBoTer/hello-ml/num"
)

type Layer struct {
	Weights num.Mat
	Biasas  num.Vector
}

func NewLayer(in, out int) *Layer {
	return &Layer{
		Weights: *num.NewMat(out, in),
		Biasas:  *num.NewVector(out),
	}
}

func (l Layer) process(in num.Vector) *num.Vector {
	return l.Weights.Mult(in).Add(l.Biasas)
}

func (l *Layer) randomize() {
	for i := range l.Biasas {
		l.Biasas[i] = rand.Float64()
	}
	height, width := l.Weights.Dim()
	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			l.Weights[i][j] = (rand.Float64() - 0.5) * 10
		}
	}
}
