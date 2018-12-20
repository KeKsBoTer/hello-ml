package decoder

import (
	"errors"
	"io/ioutil"

	"github.com/KeKsBoTer/hello-ml/num"
)

func readInt32(b []byte) int32 {
	return int32(b[3]) + (int32(b[2]) << 8) + (int32(b[1]) << 16) + (int32(b[0]) << 24)
}

type LabeledImage struct {
	Label uint8
	Image num.Vector
}

func DecodeData(imgFile, labelFile string) (*[]LabeledImage, error) {
	imgData, err := ioutil.ReadFile(imgFile)
	if err != nil {
		return nil, err
	}
	images, err := decodeImages(imgData)
	if err != nil {
		return nil, err
	}
	labelData, err := ioutil.ReadFile(labelFile)
	if err != nil {
		return nil, err
	}
	labels, err := decodeLabels(labelData)
	if err != nil {
		return nil, err
	}
	labeledImages := make([]LabeledImage, len(*images))
	for i := range labeledImages {
		labeledImages[i].Image = (*images)[i]
		labeledImages[i].Label = (*labels)[i]
	}
	return &labeledImages, nil
}

func decodeImages(buf []byte) (*[]num.Vector, error) {
	if readInt32(buf[0:]) != 2051 {
		return nil, errors.New("file must start with 0x00000803")
	}
	numImages := readInt32(buf[4:])
	images := make([]num.Vector, numImages)

	rows, columns := int(readInt32(buf[8:])), int(readInt32(buf[12:]))
	imgSize := rows * columns
	i := 0
	for offset := 0; offset < (len(buf) - imgSize); offset += imgSize {
		images[i] = make([]float64, imgSize)
		for j := 0; j < imgSize; j++ {
			images[i][j] = float64(buf[16+offset+j]) / 255
		}
		i++
	}

	return &images, nil
}

func decodeLabels(buf []byte) (*[]uint8, error) {
	if readInt32(buf[0:]) != 2049 {
		return nil, errors.New("file must start with 0x00000801")
	}
	numLabels := readInt32(buf[4:])
	labels := make([]uint8, numLabels)

	copy(labels, buf[8:])

	return &labels, nil
}
