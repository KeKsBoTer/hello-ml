package main

import (
	"math"

	"github.com/KeKsBoTer/hello-ml/decoder"
)

func main() {

	images := decoder.LabeledImages{}

	if err := images.DecodeData("data/images.data", "data/labels.data"); err != nil {
		panic(err)
	}

	nn := NewNN(sigmoid, 28*28, 16, 16, 10)
	nn.Randomize()
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
