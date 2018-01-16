package predictor

import (
	"xgboost4go-predictor/learner"
	"xgboost4go-predictor/gbm"
	"xgboost4go-predictor/util"
	"bufio"
	"xgboost4go-predictor/config"
)

type Predictor struct {
	Mparam      *PredictorModelParam
	Name_obj    string
	Name_gbm    string
	ObjFunction learner.ObjFunction
	Gbm         gbm.GradBooster
}

func NewPredictorByReader(reader bufio.Reader) (*Predictor, error) {
	return NewPredictorByConf(reader, *config.DEFAULT)
}

func NewPredictorByConf(reader bufio.Reader, configuration config.Configuration) (*Predictor, error) {
	modelReader := util.NewModelReaderByReader(reader)
	predictor := new(Predictor)
	err := predictor.readParam(modelReader)
	if err != nil {
		return predictor, err
	}
	err = predictor.initObjFunction(configuration)
	if err != nil {
		return predictor, err
	}
	err = predictor.initObjGbm()
	if err != nil {
		return predictor, err
	}
	predictor.Gbm.LoadModel(modelReader, predictor.Mparam.saved_with_pbuffer != 0)
	return predictor, nil
}

func (predictor *Predictor) readParam(reader *util.ModelReader) error {
	first4Bytes, err := reader.ReadByteArray(4)
	if err != nil {
		return err
	}
	next4Bytes, err := reader.ReadByteArray(4)
	if err != nil {
		return err
	}
	var base_score float32
	var num_feature int
	if (first4Bytes[0] == 98 && first4Bytes[1] == 105 && first4Bytes[2] == 110 && first4Bytes[3] == 102) {
		base_score = reader.AsFloat(next4Bytes)
		num_feature, err = reader.ReadUnsignedInt()
		if err != nil {
			return err
		}
	} else if (first4Bytes[0] == 0 && first4Bytes[1] == 5 && first4Bytes[2] == 95) {
		var modelType string
		if (first4Bytes[3] == 99 && next4Bytes[0] == 108 && next4Bytes[1] == 115 && next4Bytes[2] == 95) {
			modelType = "_cls_"
		} else if (first4Bytes[3] == 114 && next4Bytes[0] == 101 && next4Bytes[1] == 103 && next4Bytes[2] == 95) {
			modelType = "_reg_"
		}

		if (modelType != "") {
			temp, err := reader.ReadByteAsInt()
			if err != nil {
				return err
			}
			len := (int(next4Bytes[3]) << 8) + temp
			_, err = reader.ReadFixedString(len)
			if err != nil {
				return err
			}
			base_score, err = reader.ReadFloat()
			if err != nil {
				return err
			}
			num_feature, err = reader.ReadUnsignedInt()
			if err != nil {
				return err
			}
		} else {
			base_score = reader.AsFloat(first4Bytes)
			num_feature, err = reader.AsUnsignedInt(next4Bytes)
			if err != nil {
				return err
			}
		}
	} else {
		base_score = reader.AsFloat(first4Bytes)
		num_feature, err = reader.AsUnsignedInt(next4Bytes)
		if err != nil {
			return err
		}
	}

	predictor.Mparam, err = NewPredictorModelParam(base_score, num_feature, reader)
	if err != nil {
		return err
	}
	predictor.Name_obj, err = reader.ReadString()
	if err != nil {
		return err
	}
	predictor.Name_gbm, err = reader.ReadString()
	return err
}

func (predictor *Predictor) initObjFunction(configuration config.Configuration) error {
	var err error
	predictor.ObjFunction = configuration.ObjFunction
	if predictor.ObjFunction == nil {
		predictor.ObjFunction, err = learner.FromName(predictor.Name_obj)
	}
	return err
}

func (predictor *Predictor) initObjGbm() error {
	var err error
	predictor.ObjFunction, err = learner.FromName(predictor.Name_obj)
	if err != nil {
		return err
	}
	predictor.Gbm, err = gbm.CreateGradBooster(predictor.Name_gbm)
	if err != nil {
		return err
	}
	predictor.Gbm.SetNumClass(predictor.Mparam.num_class)
	return nil
}

func (predictor *Predictor) PredictArray(values []float32, treatsZeroAsNA bool) []float32 {
	return predictor.PredictArrayWithMargin(values, treatsZeroAsNA, false)
}

func (predictor *Predictor) PredictArrayWithMargin(values []float32, treatsZeroAsNA, output_margin bool) []float32 {
	return predictor.PredictArrayWithNtree(values, treatsZeroAsNA, output_margin, 0)
}

func (predictor *Predictor) PredictArrayWithNtree(values []float32, treatsZeroAsNA, output_margin bool, ntree_limit int) []float32 {
	preds := predictor.PredictArrayRaw(values, treatsZeroAsNA, ntree_limit)
	if output_margin {
		return preds
	} else {
		return predictor.ObjFunction.PredTransform(preds)
	}
}

func (predictor *Predictor) PredictArrayRaw(values []float32, treatsZeroAsNA bool, ntree_limit int) []float32 {
	preds := predictor.Gbm.PredictArray(values, treatsZeroAsNA, ntree_limit)
	for i := 0; i < len(preds); i++ {
		preds[i] += predictor.Mparam.base_score
	}

	return preds
}

func (predictor *Predictor) PredictArraySingle(values []float32, treatsZeroAsNA bool) float32 {
	return predictor.PredictArraySingleWithMargin(values, treatsZeroAsNA, false)
}

func (predictor *Predictor) PredictArraySingleWithMargin(values []float32, treatsZeroAsNA bool, output_margin bool) float32 {
	pred := predictor.PredictArraySingleRaw(values, treatsZeroAsNA)
	if output_margin {
		return pred
	} else {
		return predictor.ObjFunction.PredTransformSingle(pred)
	}
}

func (predictor *Predictor) PredictArraySingleRaw(values []float32, treatsZeroAsNA bool) float32 {
	temp := predictor.Gbm.PredictSingleFromArray(values, treatsZeroAsNA)
	return temp + predictor.Mparam.base_score
}

func (predictor *Predictor) PredictMap(values map[int]float32) []float32 {
	return predictor.PredictMapWithMargin(values, false)
}

func (predictor *Predictor) PredictMapWithMargin(values map[int]float32, output_margin bool) []float32 {
	return predictor.PredictMapWithNtree(values, output_margin, 0)
}

func (predictor *Predictor) PredictMapWithNtree(values map[int]float32, output_margin bool, ntree_limit int) []float32 {
	preds := predictor.PredictMapRaw(values, ntree_limit)
	if output_margin {
		return preds
	} else {
		return predictor.ObjFunction.PredTransform(preds)
	}
}

func (predictor *Predictor) PredictMapRaw(values map[int]float32, ntree_limit int) []float32 {
	preds := predictor.Gbm.PredictMap(values, ntree_limit)
	for i := 0; i < len(preds); i++ {
		preds[i] += predictor.Mparam.base_score
	}

	return preds
}

func (predictor *Predictor) PredictMapSingle(values map[int]float32) float32 {
	return predictor.PredictMapSingleWithMargin(values, false)
}

func (predictor *Predictor) PredictMapSingleWithMargin(values map[int]float32, output_margin bool) float32 {
	pred := predictor.PredictMapSingleRaw(values)
	if output_margin {
		return pred
	} else {
		return predictor.ObjFunction.PredTransformSingle(pred)
	}
}

func (predictor *Predictor) PredictMapSingleRaw(values map[int]float32) float32 {
	temp := predictor.Gbm.PredictSingleFromMap(values)
	return temp + predictor.Mparam.base_score
}

type PredictorModelParam struct {
	base_score         float32
	num_feature        int
	num_class          int
	saved_with_pbuffer int
	reserved           []int
}

func NewPredictorModelParam(base_score float32, num_feature int, reader *util.ModelReader) (*PredictorModelParam, error) {
	predictorModelParam := new(PredictorModelParam)
	var err error
	predictorModelParam.base_score = base_score
	predictorModelParam.num_feature = num_feature
	predictorModelParam.num_class, err = reader.ReadInt()
	if err != nil {
		return predictorModelParam, err
	}
	predictorModelParam.saved_with_pbuffer, err = reader.ReadInt()
	if err != nil {
		return predictorModelParam, err
	}
	predictorModelParam.reserved, err = reader.ReadIntArray(30)
	return predictorModelParam, err
}
