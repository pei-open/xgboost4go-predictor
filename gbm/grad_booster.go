package gbm

import (
	"xgboost4go-predictor/util"
	"fmt"
)

const FLOAT_32_0 = float32(0.0)

type GradBooster interface {
	SetNumClass(num_class int)
	LoadModel(modelReader *util.ModelReader, with_pbuffer bool) error
	PredictArray(values []float32, treatsZeroAsNA bool, ntree_limit int) []float32
	PredictMap(values map[int]float32, ntree_limit int) []float32
	PredictSingleFromArray(values []float32, treatsZeroAsNA bool) float32
	PredictSingleFromMap(values map[int]float32) float32
}

func CreateGradBooster(name string) (GradBooster, error) {
	if ("gbtree" == name) {
		return new(GBTree), nil
	} else if ("gblinear" == name) {
		return new(GBLinear), nil
	} else {
		return nil, fmt.Errorf("%s is not supported model.", name)
	}
}

type GBBase struct {
	NumClass int
}

func (g GBBase) SetNumClass(num_class int) {
	(&g).NumClass = num_class
}
