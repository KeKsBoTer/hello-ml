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
func (n *NeuronalNetwork) GuessNumber(image num.Vector) int {
	activations := n.process(image)
	return activations[len(activations)-1].Max()
}

func (n *NeuronalNetwork) process(input num.Vector) []num.Vector {
	activations := make([]num.Vector, len(n.layers)+1)
	activations[0] = input
	last := input
	for i, l := range n.layers {
		processed := l.process(last)
		last = *processed.Apply(sigmoid)
		activations[i+1] = last
	}
	return activations
}

func (n *NeuronalNetwork) runBatch(input []decoder.LabeledImage, learnFactor float64) float64 {
	avgCost := 0.0
	var avgGradient []Layer
	for i := 0; i < len(input); i++ {
		sample := input[i]
		activations := n.process(sample.Image)
		result := activations[len(activations)-1]
		// create expected result (1 for expected number and 0 for rest)
		expected := make([]float64, len(result))
		expected[sample.Label] = 1
		avgCost += result.Cost(expected)
		gradient := n.backprop(activations, expected)
		if i == 0 {
			avgGradient = gradient
		} else {
			for layer := range avgGradient {
				for k := range avgGradient[layer].Biasas {
					avgGradient[layer].Biasas[k] = (avgGradient[layer].Biasas[k]*float64(i) + gradient[layer].Biasas[k]) / (float64(i) + 1)
				}
				for k := range avgGradient[layer].Weights {
					for l := range avgGradient[layer].Weights[k] {
						avgGradient[layer].Weights[k][l] = (avgGradient[layer].Weights[k][l]*float64(i) + gradient[layer].Weights[k][l]) / (float64(i) + 1)
					}
				}
			}
		}
	}
	n.applyGradient(avgGradient, learnFactor)

	return avgCost / float64(len(input))
}

func (n *NeuronalNetwork) backprop(activations []num.Vector, expected num.Vector) []Layer {
	costGradient := n.emptyLayers()
	// derivation of cost to previous activations
	d_pa := *num.NewVector(len(expected))
	lastLayer := activations[len(activations)-1]
	for j := range d_pa {
		d_pa[j] = 2 * (lastLayer[j] - expected[j])
	}
	for L := len(n.layers) - 1; L >= 0; L-- {
		layer := n.layers[L]
		for j := range layer.Biasas { // amout of biasas is amount of neurons
			// calculate z of neuron j in layer L
			zLj := n.calcZ(activations, L, j)
			// calculate derivative of bias j in layer L to cost
			d_bLj := sigmoidDerivative(zLj) * d_pa[j]
			costGradient[L].Biasas[j] = d_bLj

			// calc weights
			for k := range layer.Weights[j] {
				d_wLjk := activations[L /*-1*/][k] * sigmoidDerivative(zLj) * d_pa[j]
				costGradient[L].Weights[j][k] = d_wLjk
			}
		}
		// previous Layer
		newd_pa := *num.NewVector(len(activations[L]))
		for k := range newd_pa {
			dalm1k := 0.0
			for j := range layer.Biasas {
				dalm1k += layer.Weights[j][k] * sigmoidDerivative(n.calcZ(activations, L, j)) * d_pa[j]
			}
			newd_pa[k] = dalm1k
		}
		d_pa = newd_pa
	}
	return costGradient
}

func (n *NeuronalNetwork) applyGradient(gradient []Layer, learnFactor float64) {
	for layer := range gradient {
		for k := range gradient[layer].Biasas {
			n.layers[layer].Biasas[k] -= gradient[layer].Biasas[k] * learnFactor
		}
		for k := range gradient[layer].Weights {
			for l := range gradient[layer].Weights[k] {
				n.layers[layer].Weights[k][l] -= gradient[layer].Weights[k][l] * learnFactor
			}
		}
	}
}

func (n *NeuronalNetwork) calcZ(activations []num.Vector, L int, j int) float64 {
	layer := n.layers[L]
	z := layer.Biasas[j]
	for i := range layer.Weights[j] {
		z += layer.Weights[j][i] * activations[L /*-1*/][i]
	}
	return z
}

func (n *NeuronalNetwork) emptyLayers() []Layer {
	empty := make([]Layer, len(n.layers))
	for i := range empty {
		empty[i].Biasas = *num.NewVector(len(n.layers[i].Biasas))
		empty[i].Weights = *num.NewMat(len(n.layers[i].Weights), len(n.layers[i].Weights[0]))
	}
	return empty
}

// VectorToImage takes a vector and turns it into a image with the given width
func VectorToImage(v num.Vector, imgWidth int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, imgWidth, len(v)/imgWidth))
	for i := range img.Pix {
		img.Pix[i] = uint8(v[i] * 255)
	}
	return img
}
