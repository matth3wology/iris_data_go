package data

import "math"

// Subtract two different matrices
func Subtract(a, b [][]float64) [][]float64 {
	temp := make([][]float64, len(a))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a[0]))
	}

	if len(a) == len(b) && len(a[0]) == len(b[0]) {
		for i, _ := range a {
			for j, _ := range a[i] {
				temp[i][j] = a[i][j] - b[i][j]
			}
		}
	}

	return temp
}

func Add(a, b [][]float64) [][]float64 {
	temp := make([][]float64, len(a))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a[0]))
	}

	for i, _ := range temp {
		for j, _ := range temp[0] {
			temp[i][j] = a[i][j] + b[i][j]
		}
	}

	return temp
}

// MatMul multiplys two matrices
func MatMul(x, w [][]float64) [][]float64 {

	z := make([][]float64, len(x))
	for i, _ := range z {
		z[i] = make([]float64, len(w[0]))
	}

	for i, _ := range z {
		for k, _ := range z[i] {

			for j, _ := range w {
				z[i][k] += x[i][j] * w[j][k]
			}
		}
	}

	return z
}

// MaxMin returns both max and min
func MaxMin(x [][]float64) (float64, float64) {

	var max float64 = 0.0
	var min float64 = 0.0

	for i, _ := range x {
		for j, _ := range x[i] {

			if x[i][j] > max {
				max = x[i][j]
			}

			if x[i][j] < min {
				min = x[i][j]
			}
		}
	}

	return max, min
}

// Normalize the data
func Normalize(x [][]float64) [][]float64 {

	temp := make([][]float64, len(x))
	for i, _ := range temp {
		temp[i] = make([]float64, len(x[0]))
	}

	max, min := MaxMin(x)

	// z = (x-min)/(max-min)
	for i, _ := range x {
		for j, _ := range x[i] {
			temp[i][j] = (x[i][j] - min) / (max - min)
		}
	}

	return temp
}

// Error of two matrices
func Error(a, y [][]float64) float64 {
	var error float64 = 0.0

	for i, _ := range a {
		for j, _ := range a[i] {
			error += (0.5 * ((a[i][j] - y[i][j]) * (a[i][j] - y[i][j])))
		}
	}

	return error
}

func Square(a [][]float64) [][]float64 {
	temp := make([][]float64, len(a))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a[0]))
	}

	for i, _ := range temp {
		for j, _ := range temp[0] {
			temp[i][j] = a[i][j] * a[i][j]
		}
	}

	return temp
}

// Sigmoid logistic function
func Sigmoid(x [][]float64) [][]float64 {
	temp := make([][]float64, len(x))
	for i, _ := range temp {
		temp[i] = make([]float64, len(x[0]))
	}

	// sig(z) = {1 / (1 + e ^ -z)}
	for i, _ := range temp {
		for j, _ := range temp[i] {
			temp[i][j] = 1.0 / (1.0 + math.Exp(-1.0*x[i][j]))
		}
	}

	return temp
}

func SigmoidDeriv(z [][]float64) [][]float64 {

	//sig_(z) = sig(z) - sig(z)^2
	sig := Sigmoid(z)
	sig2 := Square(sig)
	derivative := Subtract(sig, sig2)

	return derivative
}

func Dot(a, b [][]float64) [][]float64 {

	temp := make([][]float64, len(a))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a[0]))
	}

	for i, _ := range temp {
		for j, _ := range temp[i] {
			temp[i][j] = a[i][j] * b[i][j]
		}
	}

	return temp
}

func Transform(a [][]float64) [][]float64 {
	temp := make([][]float64, len(a[0]))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a))
	}

	for i, _ := range temp {
		for j, _ := range temp[i] {
			temp[i][j] = a[j][i]
		}
	}

	return temp
}

func MatScalar(mat [][]float64, scal float64) [][]float64 {
	temp := make([][]float64, len(mat))
	for i, _ := range temp {
		temp[i] = make([]float64, len(mat[0]))
	}

	for i, _ := range temp {
		for j, _ := range temp[i] {
			temp[i][j] = mat[i][j] * scal
		}
	}

	return temp
}

func AddBias(a [][]float64) [][]float64 {
	temp := make([][]float64, len(a))
	for i, _ := range temp {
		temp[i] = make([]float64, len(a[0]))
	}

	for i, _ := range temp {
		for j, _ := range temp[0] {
			temp[i][j] = a[i][j] + 1.0
		}
	}

	return temp
}
