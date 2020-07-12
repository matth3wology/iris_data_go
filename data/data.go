package data

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type TrainingData struct {
	Data         [][]float64
	Labels       []string
	LabelsOneHot [][]float64
}

func (t *TrainingData) Load(filename string) {

	csvFile, err := os.Open("iris.csv")
	if err != nil {
		log.Fatalln("Error opening file", err)
	}

	reader := csv.NewReader(csvFile)

	dataset, err := reader.ReadAll()
	if err != nil {
		log.Fatalln("Error reading bytes", err)
	}

	for i, row := range dataset {

		if i > 0 {
			sepal_length, err := strconv.ParseFloat(row[0], 64)
			if err != nil {
				log.Fatalln("Error converting string to float", err)
			}
			sepal_width, err := strconv.ParseFloat(row[1], 64)
			if err != nil {
				log.Fatalln("Error converting string to float", err)
			}
			petal_length, err := strconv.ParseFloat(row[2], 64)
			if err != nil {
				log.Fatalln("Error converting string to float", err)
			}
			petal_width, err := strconv.ParseFloat(row[3], 64)
			if err != nil {
				log.Fatalln("Error converting string to float", err)
			}

			rowData := []float64{sepal_length, sepal_width, petal_length, petal_width}

			t.Data = append(t.Data, rowData)
			t.Labels = append(t.Labels, row[4])

			t.OneHot()
		}
	}
}

func (t *TrainingData) DisplayData() {
	for _, row := range t.Data {
		fmt.Println(row)
	}
}

func (t *TrainingData) DisplayLabels() {
	for _, row := range t.Labels {
		fmt.Println(row)
	}
}

func (t *TrainingData) RandomizeDataset() {
	rand.Seed(time.Now().UnixNano())

	if len(t.Labels) == len(t.Data) {
		rand.Shuffle(len(t.Data), func(i, j int) {
			t.Data[i], t.Data[j] = t.Data[j], t.Data[i]
			t.Labels[i], t.Labels[j] = t.Labels[j], t.Labels[i]
			t.LabelsOneHot[i], t.LabelsOneHot[j] = t.LabelsOneHot[j], t.LabelsOneHot[i]
		})
	}
}

func (t *TrainingData) Batch(number int) ([][]float64, []string) {
	return t.Data[:number], t.Labels[:number]
}

func (t *TrainingData) BatchNormalize(number int) ([][]float64, [][]float64) {
	return Normalize(t.Data[:number]), t.LabelsOneHot[:number]
}

func (t *TrainingData) OneHot() {

	temp := make([][]float64, len(t.Labels))
	for i, _ := range temp {
		temp[i] = make([]float64, 3)
	}

	if len(t.Data) > 0 && t.Data != nil && len(t.Labels) > 0 && t.Labels != nil {

		row := make([]float64, 3)

		for i, label := range t.Labels {

			switch label {

			case "setosa":
				row = []float64{1.0, 0.0, 0.0}
				temp[i] = row

			case "versicolor":
				row = []float64{0.0, 1.0, 0.0}
				temp[i] = row

			case "virginica":
				row = []float64{0.0, 0.0, 1.0}
				temp[i] = row
			}

		}
	}

	t.LabelsOneHot = temp
}
