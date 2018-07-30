package main

import (
	"io/ioutil"
	"log"
	"reflect"
	"testing"
)

func TestNeuronalNetwork_GobEncodeDecode(t *testing.T) {

	file, err := ioutil.TempFile("", "test.gob")
	if err != nil {
		t.Fatal("cannot create temp file:", err)
	}
	//defer os.Remove(file.Name())

	nn := NewNN(sigmoid, 28*28, 16, 16, 10)
	nn.Randomize()
	out, err := nn.GobEncode()
	if err != nil {
		t.Fatal("cannot create encode layers:", err)
	}

	if _, err := file.Write(out); err != nil {
		t.Fatal("cannot write to temp file:", err)
	}

	data, err := ioutil.ReadFile("data.gob")
	if err != nil {
		log.Fatalln(err)
	}
	loadedNN := NeuronalNetwork{}
	loadedNN.GobDecode(data)

	if !reflect.DeepEqual(nn.layers, loadedNN.layers) {
		t.Errorf("Decoded file has different values than original nn")
	}
}
