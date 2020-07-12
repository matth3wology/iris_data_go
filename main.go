package main

import (
	"fmt"
	"test/data"
)

func main() {

	// Setup the Model
	var trainingSet data.TrainingData

	trainingSet.Load("./res/iris.csv")

	trainingSet.RandomizeDataset()

	x, y := trainingSet.BatchNormalize(25)

	weights := [][]float64{
		{0.2, 0.1, 0.5},
		{0.2, 0.9, 0.4},
		{0.1, 0.5, 0.5},
		{0.2, 0.1, 0.5},
	}

	for i := 0; i < 1000000; i++ {

		// Feed Forward
		z := data.AddBias(data.MatMul(x, weights))

		a := data.Sigmoid(z)

		// Check Error
		error := data.Error(a, y)

		fmt.Printf("Iteration: %d, Error: %0.4f \n", i, error)

		// Gradient Update
		delta := data.Dot(data.Subtract(y, a), data.SigmoidDeriv(z))

		xT := data.Transform(x)

		newWeights := data.MatMul(xT, delta)

		weights = data.Add(weights, data.MatScalar(newWeights, 0.35))

	}

}
