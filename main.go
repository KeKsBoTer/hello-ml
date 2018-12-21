package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/KeKsBoTer/hello-ml/decoder"
)

func main() {

	if len(os.Args) < 1 {
		fmt.Println("run with 'train' or 'test' as argument")
		return
	}
	mode := os.Args[1]

	rand.Seed(time.Now().UnixNano())
	images, err := decoder.DecodeData("data/images.data", "data/labels.data")

	if err != nil {
		panic(err)
	}
	nn := NewNN(28*28, 16, 16, 10)

	b, err := ioutil.ReadFile("models/model_1000.gob")
	if err != nil {
		fmt.Println(err)
		return
	}
	nn.GobDecode(b)

	switch mode {
	case "train":
		fmt.Println("runing 10000 epochs")
		for i := 0; i < 10000; i++ {
			out := nn.runBatch((*images)[0:100], 0.1)
			fmt.Printf("Step %d, Cost:  %f\n", i, out)
			perm := rand.Perm(len(*images))
			for i, v := range perm {
				(*images)[v], (*images)[i] = (*images)[i], (*images)[v]
			}
			if i%1000 == 0 {
				data, err := nn.GobEncode()
				if err != nil {
					fmt.Println(err)
				} else {
					err := ioutil.WriteFile(fmt.Sprintf("models/model_%d.gob", i), data, 0x666)
					if err != nil {
						fmt.Println(err)
					}
				}
			}
		}
	case "test":
		fmt.Println("running 100 random test numbers")
		accuracy := make([]int, 100)
		for i := 0; i < 100; i++ {
			guess := nn.GuessNumber((*images)[i].Image)
			fmt.Printf("image is a %d, guessed %d\n", (*images)[i].Label, guess)
			if uint8(guess) == (*images)[i].Label {
				accuracy[i] = 1
			}
		}
		sum := 0
		for _, v := range accuracy {
			sum += v
		}
		fmt.Println("accuracy:", float64(sum)/100)
	default:
		fmt.Println("unknown mode. run with 'train' or 'test' as argument")
		return
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}
