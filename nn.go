package main

import (
	"bytes"
	"encoding/gob"

	"github.com/KeKsBoTer/hello-ml/num"
)

type ActivationFunction func(float64) float64

type NeuronalNetwork struct {
	activation ActivationFunction
	layers     []Layer
}

func NewNN(afunc ActivationFunction, layerSizes ...int) *NeuronalNetwork {
	nn := NeuronalNetwork{
		activation: afunc,
		layers:     make([]Layer, len(layerSizes)),
	}
	for i := 0; i < len(nn.layers)-1; i++ {
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

func (n *NeuronalNetwork) Process(input num.Vector) *num.Vector {
	last := input
	for i := range n.layers {
		last = *n.layers[i].process(last).Apply(n.activation)
	}
	return &last
}
