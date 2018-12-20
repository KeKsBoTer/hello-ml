package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/KeKsBoTer/hello-ml/decoder"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	images, err := decoder.DecodeData("data/images.data", "data/labels.data")

	if err != nil {
		panic(err)
	}
	nn := NewNN(28*28, 16, 16, 10)
	nn.Randomize()

	out := nn.runBatch(images)
	fmt.Println("Cost: ", out)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}
