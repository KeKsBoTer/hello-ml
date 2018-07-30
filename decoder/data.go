package decoder

import (
	"errors"
	"image"
	"io/ioutil"
)

func readInt32(b []byte) int32 {
	return int32(b[3]) + (int32(b[2]) << 8) + (int32(b[1]) << 16) + (int32(b[0]) << 24)
}

type LabeledImages struct {
	labels []uint8
	images []image.Gray
}

func (li LabeledImages) Get(i int) (uint8, image.Gray) {
	return li.labels[i], li.images[i]
}

func (li LabeledImages) Length() int {
	return len(li.images)
}

func (li *LabeledImages) DecodeData(imgFile, labelFile string) error {
	imgData, err := ioutil.ReadFile(imgFile)
	if err != nil {
		return err
	}
	if err = li.decodeImages(imgData); err != nil {
		return err
	}
	labelData, err := ioutil.ReadFile(labelFile)
	if err != nil {
		return err
	}
	if err = li.decodeLabels(labelData); err != nil {
		return err
	}
	return nil
}

func (li *LabeledImages) decodeImages(buf []byte) error {
	if readInt32(buf[0:]) != 2051 {
		return errors.New("file must start with 0x00000803")
	}
	numImages := readInt32(buf[4:])
	li.images = make([]image.Gray, numImages)

	rows, columns := int(readInt32(buf[8:])), int(readInt32(buf[12:]))
	imgSize := rows * columns
	i := 0
	for offset := 0; offset < (len(buf) - imgSize); offset += imgSize {
		li.images[i].Rect = image.Rect(0, 0, rows, columns)
		li.images[i].Pix = make([]uint8, imgSize)
		li.images[i].Stride = columns
		copy(li.images[i].Pix, buf[(16+offset):(16+offset+imgSize)])
		i++
	}

	return nil
}

func (li *LabeledImages) decodeLabels(buf []byte) error {
	if readInt32(buf[0:]) != 2049 {
		return errors.New("file must start with 0x00000801")
	}
	numLabels := readInt32(buf[4:])
	li.labels = make([]uint8, numLabels)

	copy(li.labels, buf[8:])

	return nil
}
