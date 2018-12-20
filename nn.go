package main

import (
	"bytes"
	"encoding/gob"
	"image"

	"github.com/KeKsBoTer/hello-ml/decoder"
	"github.com/KeKsBoTer/hello-ml/num"
)

type ActivationFunction func(float64) float64

type NeuronalNetwork struct {
	layers []Layer
}

func NewNN(layerSizes ...int) *NeuronalNetwork {
	nn := NeuronalNetwork{
		layers: make([]Layer, len(layerSizes)-1),
	}
	for i := 0; i < len(nn.layers); i++ {
		nn.layers[i] = *NewLayer(layerSizes[i], layerSizes[i+1])
	}
	return &nn
}

func (n *NeuronalNetwork) Randomize() {
	for i := range n.layers {
		n.layers[i].randomize()
	}
}

func (n *NeuronalNetwork) GobEncode() (buf []byte, err error) {
	w := bytes.NewBuffer(buf)
	encoder := gob.NewEncoder(w)
	err = encoder.Encode(n.layers)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), err
}

func (n *NeuronalNetwork) GobDecode(buf []byte) (err error) {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)
	err = decoder.Decode(&n.layers)
	if err != nil {
		return err
	}
	return err
}

// GuessNumber guesses the number in the image
func (n *NeuronalNetwork) GuessNumber(image *decoder.LabeledImage) int {
	return n.process(image.Image).Max()
}

func (n *NeuronalNetwork) process(input num.Vector) *num.Vector {
	last := input
	for _, l := range n.layers {
		processed := l.process(last)
		last = *processed.Apply(sigmoid)
	}
	return &last
}

func (n *NeuronalNetwork) runBatch(input *[]decoder.LabeledImage) float64 {
	avgCost := 0.0
	for i := 0; i < len(*input); i++ {
		sample := (*input)[i]
		result := n.process(sample.Image)
		// create expected result (1 for expected number and 0 for rest)
		expected := make([]float64, len(*result))
		expected[sample.Label] = 1
		avgCost += result.Cost(expected)
		n.backprop(*result, expected)
	}
	return avgCost / float64(len(*input))
}

func (n *NeuronalNetwork) backprop(out, expected num.Vector) {
	/*
		nabla_b := make([]num.Vector, len(n.layers))
		nabla_w := make([]num.Vector, len(n.layers))
		for i := range nabla_b {
			nabla_b[i] = *num.NewVector(len(n.layers[i].Biasas))
			nabla_w[i] = *num.NewVector(len(n.layers[i].Weights))
		}

		activation := out
		activations := []num.Vector{out}
		var zs []*num.Vector

		for _, l := range n.layers {
			z := l.process(activation)
			zs = append(zs, z)
			activation = *z.Apply(sigmoid)
			activations = append(activations, activation)
		}

		costs := *n.costDerivative(activations[len(activations)-1], expected)
		sigmoids := *zs[len(zs)-1].Apply(sigmoidDerivative)
		delta := *num.NewVector(len(costs))
		for i := range delta {
			delta[i] = costs[i] * sigmoids[i]
		}

		nabla_b[len(nabla_b)-1] = delta
		nabla_w[len(nabla_w)-1] = *delta.Dot(activations[len(activations)-2])
	*/
}

func (n *NeuronalNetwork) costDerivative(out, y num.Vector) *num.Vector {
	return &out
}

// VectorToImage takes a vector and turns it into a image with the given width
func VectorToImage(v num.Vector, imgWidth int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, imgWidth, len(v)/imgWidth))
	for i := range img.Pix {
		img.Pix[i] = uint8(v[i] * 255)
	}
	return img
}
