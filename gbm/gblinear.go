package gbm

import (
	"xgboost4go-predictor/util"
	"xgboost4go-predictor/math"
)

type GBLinear struct {
	GBBase
	mparam  *GBLinearParam
	weights []float32
}

func (gbLinear *GBLinear) LoadModel(reader *util.ModelReader, ignored_with_pbuffer bool) error {
	var err error
	gbLinear.mparam, err = newGBLinearParam(reader)
	if err != nil {
		return err
	}
	_, err = reader.ReadInt()
	if err != nil {
		return err
	}
	gbLinear.weights, err = reader.ReadFloatArray((gbLinear.mparam.num_feature + 1) * gbLinear.mparam.num_output_group)
	return err
}

func (gbLinear *GBLinear) PredictArray(values []float32, treatsZeroAsNA bool, ntree_limit int) []float32 {
	preds := make([]float32, gbLinear.mparam.num_output_group)
	for gid := 0; gid < gbLinear.mparam.num_output_group; gid++ {
		preds[gid] = gbLinear.PredFromArray(values, treatsZeroAsNA, gid)
	}

	return preds
}

func (gbLinear *GBLinear) PredictMap(values map[int]float32, ntree_limit int) []float32 {
	preds := make([]float32, gbLinear.mparam.num_output_group)
	for gid := 0; gid < gbLinear.mparam.num_output_group; gid++ {
		preds[gid] = gbLinear.PredFromMap(values, gid)
	}

	return preds
}

func (gbLinear *GBLinear) PredictSingleFromArray(values []float32, treatsZeroAsNA bool) float32 {
	if (gbLinear.mparam.num_output_group != 1) {
		return math.NAN
	} else {
		return gbLinear.PredFromArray(values, treatsZeroAsNA, 0)
	}
}

func (gbLinear *GBLinear) PredictSingleFromMap(values map[int]float32) float32 {
	if (gbLinear.mparam.num_output_group != 1) {
		return math.NAN
	} else {
		return gbLinear.PredFromMap(values, 0)
	}
}

func (gbLinear *GBLinear) PredFromArray(value []float32, treatsZeroAsNA bool, gid int) float32 {
	psum := gbLinear.Bias(gid)
	for fid := 0; fid < gbLinear.mparam.num_feature; fid++ {
		if len(value) > fid {
			featValue := value[fid]
			if !treatsZeroAsNA || featValue != 0 {
				psum += featValue * gbLinear.Weight(fid, gid)
			}
		}
	}

	return psum
}

func (gbLinear *GBLinear) PredFromMap(values map[int]float32, gid int) float32 {
	psum := gbLinear.Bias(gid)
	for fid := 0; fid < gbLinear.mparam.num_feature; fid++ {
		featValue, ok := values[fid]
		if !ok {
			psum += featValue * gbLinear.Weight(fid, gid)
		}
	}

	return psum
}

func (gbLinear *GBLinear) Weight(fid, gid int) float32 {
	return gbLinear.weights[fid*gbLinear.mparam.num_output_group+gid]
}

func (gbLinear *GBLinear) Bias(gid int) float32 {
	return gbLinear.weights[gbLinear.mparam.num_feature*gbLinear.mparam.num_output_group+gid]
}

type GBLinearParam struct {
	num_feature      int
	num_output_group int
	reserved         []int
}

func newGBLinearParam(reader *util.ModelReader) (*GBLinearParam, error) {
	gbLinearParam := new(GBLinearParam)
	var err error
	gbLinearParam.num_feature, err = reader.ReadInt()
	if err != nil {
		return gbLinearParam, err
	}
	gbLinearParam.num_output_group, err = reader.ReadInt()
	if err != nil {
		return gbLinearParam, err
	}
	gbLinearParam.reserved, err = reader.ReadIntArray(32)
	if err != nil {
		return gbLinearParam, err
	}
	_, err = reader.ReadInt()
	return gbLinearParam, err
}
