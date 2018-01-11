package xgboost4go_predictor

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
	modelReader := util.NewModelReaderByReader(reader);
	predictor := new(Predictor)
	err := predictor.readParam(modelReader);
	if err != nil {
		return predictor, err
	}
	err = predictor.initObjFunction(configuration);
	if err != nil {
		return predictor, err
	}
	err = predictor.initObjGbm();
	if err != nil {
		return predictor, err
	}
	predictor.Gbm.LoadModel(modelReader, predictor.Mparam.saved_with_pbuffer != 0);
	return predictor, nil
}

func (predictor *Predictor) readParam(reader *util.ModelReader) error {
	first4Bytes, err := reader.ReadByteArray(4);
	if err != nil {
		return err
	}
	next4Bytes, err := reader.ReadByteArray(4);
	if err != nil {
		return err
	}
	var base_score float32
	var num_feature int
	if (first4Bytes[0] == 98 && first4Bytes[1] == 105 && first4Bytes[2] == 110 && first4Bytes[3] == 102) {
		base_score = reader.AsFloat(next4Bytes);
		num_feature, err = reader.ReadUnsignedInt();
		if err != nil {
			return err
		}
	} else if (first4Bytes[0] == 0 && first4Bytes[1] == 5 && first4Bytes[2] == 95) {
		var modelType string
		if (first4Bytes[3] == 99 && next4Bytes[0] == 108 && next4Bytes[1] == 115 && next4Bytes[2] == 95) {
			modelType = "_cls_";
		} else if (first4Bytes[3] == 114 && next4Bytes[0] == 101 && next4Bytes[1] == 103 && next4Bytes[2] == 95) {
			modelType = "_reg_";
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
			base_score, err = reader.ReadFloat();
			if err != nil {
				return err
			}
			num_feature, err = reader.ReadUnsignedInt();
			if err != nil {
				return err
			}
		} else {
			base_score = reader.AsFloat(first4Bytes);
			num_feature, err = reader.AsUnsignedInt(next4Bytes);
			if err != nil {
				return err
			}
		}
	} else {
		base_score = reader.AsFloat(first4Bytes);
		num_feature, err = reader.AsUnsignedInt(next4Bytes);
		if err != nil {
			return err
		}
	}

	predictor.Mparam, err = NewPredictorModelParam(base_score, num_feature, reader)
	if err != nil {
		return err
	}
	predictor.Name_obj, err = reader.ReadString();
	if err != nil {
		return err
	}
	predictor.Name_gbm, err = reader.ReadString();
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

func (predictor *Predictor) Predict(feat util.FVec) []float32 {
	return predictor.PredictWithMargin(feat, false)
}

func (predictor *Predictor) PredictWithMargin(feat util.FVec, output_margin bool) []float32 {
	return predictor.PredictWithNtree(feat, output_margin, 0)
}

func (predictor *Predictor) PredictWithNtree(feat util.FVec, output_margin bool, ntree_limit int) []float32 {
	preds, err := predictor.predictRaw(feat, ntree_limit)
	if err != nil {
		return preds
	}
	if output_margin {
		return predictor.ObjFunction.PredTransform(preds)
	} else {
		return preds
	}
}

func (predictor *Predictor) predictRaw(feat util.FVec, ntree_limit int) ([]float32, error) {
	preds, err := predictor.Gbm.Predict(feat, ntree_limit)
	if err != nil {
		return preds, err
	}
	for i := 0; i < len(preds); i++ {
		preds[i] += predictor.Mparam.base_score
	}

	return preds, nil
}

func (predictor *Predictor) PredictSingle(feat util.FVec) (float32, error) {
	return predictor.PredictSingleWithMargin(feat, false)
}

func (predictor *Predictor) PredictSingleWithMargin(feat util.FVec, output_margin bool) (float32, error) {
	return predictor.PredictSingleWithNtree(feat, output_margin, 0)
}

func (predictor *Predictor) PredictSingleWithNtree(feat util.FVec, output_margin bool, ntree_limit int) (float32, error) {
	pred, err := predictor.PredictSingleRaw(feat, ntree_limit)
	if err != nil {
		return pred, err
	}
	if output_margin {
		return predictor.ObjFunction.PredTransformSingle(pred)
	} else {
		return pred, nil
	}
}

func (predictor *Predictor) PredictSingleRaw(feat util.FVec, ntree_limit int) (float32, error) {
	temp, err := predictor.Gbm.PredictSingle(feat, ntree_limit)
	if err != nil {
		return 0, err
	}
	return temp + predictor.Mparam.base_score, nil
}

func (predictor *Predictor) PredictLeaf(feat util.FVec) ([]int, error) {
	return predictor.PredictLeafWithNtree(feat, 0)
}

func (predictor *Predictor) PredictLeafWithNtree(feat util.FVec, ntree_limit int) ([]int, error) {
	return predictor.Gbm.PredictLeaf(feat, ntree_limit)
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
