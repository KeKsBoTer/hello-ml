package num

type Mat [][]float64

func NewMat(i, j int) *Mat {
	m := make(Mat, i)
	for i := range m {
		m[i] = make([]float64, j)
	}
	return &m
}

func (m Mat) Dim() (int, int) {
	i, j := 0, 0
	i = len(m)
	if i > 0 {
		j = len(m[0])
	}
	return i, j
}

func (m Mat) Mult(v Vector) *Vector {
	height, _ := m.Dim()
	r := *NewVector(height)
	for i, p := range m {
		r[i] = v.Dot(Vector(p))
	}
	return &r
}
