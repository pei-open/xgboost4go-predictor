package learner

import (
	"fmt"
	"xgboost4go-predictor/math"
)

var FUNCTIONS = make(map[string]ObjFunction)

func init() {
	Register("rank:pairwise", new(DefaultObjFunction))
	Register("binary:logistic", new(RegLossObjLogistic))
	Register("binary:logitraw", new(DefaultObjFunction))
	Register("multi:softmax", new(SoftmaxMultiClassObjClassify))
	Register("multi:softprob", new(SoftmaxMultiClassObjProb))
	Register("reg:linear", new(DefaultObjFunction))
}

func FromName(name string) (ObjFunction, error) {
	objFunction, ok := FUNCTIONS[name]
	if !ok {
		return objFunction, fmt.Errorf("%s is not supported objective function.", name)
	} else {
		return objFunction, nil
	}
}

func UseFastMathExp(useJafama bool) {
	if (useJafama) {
		Register("binary:logistic", new(RegLossObjLogisticJafama))
		Register("multi:softprob", new(SoftmaxMultiClassObjProbJafama))
	} else {
		Register("binary:logistic", new(RegLossObjLogistic))
		Register("multi:softprob", new(SoftmaxMultiClassObjProb))
	}
}

func Register(name string, objFunction ObjFunction) {
	FUNCTIONS[name] = objFunction
}

type ObjFunction interface {
	PredTransformSingle(pred float32) (float32, error)
	PredTransform(preds []float32) []float32
}

type DefaultObjFunction struct {
}

func (dof DefaultObjFunction) PredTransformSingle(pred float32) (float32, error) {
	return pred, nil
}

func (dof DefaultObjFunction) PredTransform(preds []float32) []float32 {
	return preds
}

type SoftmaxMultiClassObjProb struct {
}

func (smcop SoftmaxMultiClassObjProb) PredTransformSingle(pred float32) (float32, error) {
	return 0, fmt.Errorf("function not supported in SoftmaxMultiClassObjProb")
}

func (smcop SoftmaxMultiClassObjProb) PredTransform(preds []float32) []float32 {
	max := preds[0]
	for i := 1; i < len(preds); i++ {
		max = math.MaxFloat32(preds[i], max)
	}

	sum := float32(0.0)
	for i := 0; i < len(preds); i++ {
		preds[i] = math.ExpFloat32(preds[i] - max);
		sum += preds[i]
	}

	for i := 0; i < len(preds); i++ {
		preds[i] /= sum
	}

	return preds
}

type SoftmaxMultiClassObjProbJafama struct {
}

func (smcopj SoftmaxMultiClassObjProbJafama) PredTransformSingle(pred float32) (float32, error) {
	return 0, fmt.Errorf("function not supported in SoftmaxMultiClassObjProbJafama")
}

func (smcopj SoftmaxMultiClassObjProbJafama) PredTransform(preds []float32) []float32 {
	max := preds[0]
	for i := 1; i < len(preds); i++ {
		max = math.MaxFloat32(preds[i], max)
	}

	sum := float32(0.0)
	for i := 0; i < len(preds); i++ {
		preds[i] = math.ExpFloat32(preds[i] - max)
		sum += preds[i]
	}

	for i := 0; i < len(preds); i++ {
		preds[i] /= sum
	}

	return preds
}

type SoftmaxMultiClassObjClassify struct {
}

func (smcoc SoftmaxMultiClassObjClassify) PredTransformSingle(pred float32) (float32, error) {
	return 0, fmt.Errorf("function not supported in SoftmaxMultiClassObjClassify")
}

func (smcoc SoftmaxMultiClassObjClassify) PredTransform(preds []float32) []float32 {
	maxIndex := 0
	max := preds[0]
	for i := 0; i < len(preds); i++ {
		if (max < preds[i]) {
			maxIndex = i
			max = preds[i]
		}
	}

	return append(make([]float32, 1), float32(maxIndex))
}

type RegLossObjLogistic struct {
}

func (rlol RegLossObjLogistic) PredTransformSingle(pred float32) (float32, error) {
	return rlol.Sigmoid(pred), nil
}

func (rlol RegLossObjLogistic) PredTransform(preds []float32) []float32 {
	for i := 0; i < len(preds); i++ {
		preds[i] = rlol.Sigmoid(preds[i])
	}

	return preds
}

func (rlol RegLossObjLogistic) Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.ExpFloat32(-x))
}

type RegLossObjLogisticJafama struct {
}

func (rlolj RegLossObjLogisticJafama) PredTransformSingle(pred float32) (float32, error) {
	return rlolj.Sigmoid(pred), nil
}

func (rlolj RegLossObjLogisticJafama) PredTransform(preds []float32) []float32 {
	for i := 0; i < len(preds); i++ {
		preds[i] = rlolj.Sigmoid(preds[i])
	}

	return preds
}

func (rlolj RegLossObjLogisticJafama) Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.ExpFloat32(-x))
}
