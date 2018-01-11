package gbm

import (
	"xgboost4go-predictor/util"
	"xgboost4go-predictor/tree"
	"fmt"
	"github.com/chewxy/math32"
)

type GradBooster interface {
	SetNumClass(num_class int)
	LoadModel(modelReader *util.ModelReader, var2 bool) error
	Predict(fvec util.FVec, var2 int) ([]float32, error)
	PredictSingle(fvec util.FVec, var2 int) (float32, error)
	PredictLeaf(fvec util.FVec, var2 int) ([]int, error)
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

type GBTree struct {
	GBBase
	mparam      *GBTreeParam
	trees       []*tree.RegTree
	tree_info   []int
	_groupTrees [][]*tree.RegTree
}

func (gbTree *GBTree) LoadModel(reader *util.ModelReader, with_pbuffer bool) error {
	var err error
	gbTree.mparam, err = newGBTreeParam(reader)
	if err != nil {
		return err
	}

	gbTree.trees = make([]*tree.RegTree, gbTree.mparam.num_trees)
	for i := 0; i < gbTree.mparam.num_trees; i++ {
		gbTree.trees[i] = new(tree.RegTree)
		err = gbTree.trees[i].LoadModel(reader)
		if err != nil {
			return err
		}
	}

	if gbTree.mparam.num_trees != 0 {
		gbTree.tree_info, err = reader.ReadIntArray(gbTree.mparam.num_trees)
		if err != nil {
			return err
		}
	}

	if gbTree.mparam.num_pbuffer != 0 && with_pbuffer {
		reader.Skip(4 * int(gbTree.mparam.PredBufferSize()))
		reader.Skip(4 * int(gbTree.mparam.PredBufferSize()))
	}

	gbTree._groupTrees = make([][]*tree.RegTree, gbTree.mparam.num_output_group)
	for i := 0; i < gbTree.mparam.num_output_group; i++ {
		treeCount := 0
		for j := 0; j < len(gbTree.tree_info); j++ {
			if (gbTree.tree_info[j] == i) {
				treeCount++
			}
		}

		gbTree._groupTrees[i] = make([]*tree.RegTree, treeCount)
		treeCount = 0
		for j := 0; j < len(gbTree.tree_info); j++ {
			if (gbTree.tree_info[j] == i) {
				gbTree._groupTrees[i][treeCount] = gbTree.trees[j]
				treeCount++
			}
		}
	}
	return err
}

func (gbTree *GBTree) Predict(feat util.FVec, ntree_limit int) ([]float32, error) {
	preds := make([]float32, gbTree.mparam.num_output_group)
	for gid := 0; gid < gbTree.mparam.num_output_group; gid++ {
		preds[gid] = gbTree.Pred(feat, gid, 0, ntree_limit);
	}

	return preds, nil
}

func (gbTree *GBTree) PredictSingle(feat util.FVec, ntree_limit int) (float32, error) {
	if (gbTree.mparam.num_output_group != 1) {
		return 0, fmt.Errorf("Can't invoke predictSingle() because this model outputs multiple values: %d", gbTree.mparam.num_output_group)
	} else {
		return gbTree.Pred(feat, 0, 0, ntree_limit), nil
	}
}

func (gbTree *GBTree) Pred(feat util.FVec, bst_group, root_index, ntree_limit int) float32 {
	trees := gbTree._groupTrees[bst_group]
	treeleft := ntree_limit
	if ntree_limit == 0 {
		treeleft = len(trees)
	}
	psum := float32(0.0)
	for i := 0; i < treeleft; i++ {
		psum += trees[i].GetLeafValueFloat(feat, root_index)
	}

	return psum
}

func (gbTree *GBTree) PredictLeaf(feat util.FVec, ntree_limit int) ([]int, error) {
	return gbTree.PredPath(feat, 0, ntree_limit), nil
}

func (gbTree *GBTree) PredPath(feat util.FVec, root_index, ntree_limit int) []int {
	var treeleft int
	if ntree_limit == 0 {
		treeleft = len(gbTree.trees)
	} else {
		treeleft = ntree_limit
	}
	leafIndex := make([]int, treeleft)
	for i := 0; i < treeleft; i++ {
		leafIndex[i] = gbTree.trees[i].GetLeafIndex(feat, root_index)
	}

	return leafIndex
}

type GBTreeParam struct {
	num_trees        int
	num_roots        int
	num_feature      int
	num_pbuffer      int64
	num_output_group int
	size_leaf_vector int
	reserved         []int
}

func (gbTreeParam *GBTreeParam) predBufferSize() int64 {
	return int64(gbTreeParam.num_output_group) * gbTreeParam.num_pbuffer * int64(gbTreeParam.size_leaf_vector+1)
}

func newGBTreeParam(reader *util.ModelReader) (*GBTreeParam, error) {
	gbTreeParam := new(GBTreeParam)
	var err error
	gbTreeParam.num_trees, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.num_roots, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.num_feature, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	_, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.num_pbuffer, err = reader.ReadInt64()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.num_output_group, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.size_leaf_vector, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	gbTreeParam.reserved, err = reader.ReadIntArray(31)
	if err != nil {
		return gbTreeParam, err
	}
	_, err = reader.ReadInt()
	if err != nil {
		return gbTreeParam, err
	}
	return gbTreeParam, nil
}

func (gbTreeParam *GBTreeParam) PredBufferSize() int64 {
	return int64(gbTreeParam.num_output_group) * gbTreeParam.num_pbuffer * int64(gbTreeParam.size_leaf_vector+1)
}

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

func (gbLinear *GBLinear) Predict(feat util.FVec, ntree_limit int) ([]float32, error) {
	preds := make([]float32, gbLinear.mparam.num_output_group)
	for gid := 0; gid < gbLinear.mparam.num_output_group; gid++ {
		preds[gid] = gbLinear.Pred(feat, gid)
	}

	return preds, nil
}

func (gbLinear *GBLinear) PredictSingle(feat util.FVec, ntree_limit int) (float32, error) {
	if (gbLinear.mparam.num_output_group != 1) {
		return 0, fmt.Errorf("Can't invoke predictSingle() because this model outputs multiple values: %d", gbLinear.mparam.num_output_group)
	} else {
		return gbLinear.Pred(feat, 0), nil
	}
}

func (gbLinear *GBLinear) Pred(feat util.FVec, gid int) float32 {
	psum := gbLinear.Bias(gid);
	for fid := 0; fid < gbLinear.mparam.num_feature; fid++ {
		featValue := feat.Fvalue(fid);
		if (!math32.IsNaN(featValue)) {
			psum += featValue * gbLinear.Weight(fid, gid);
		}
	}

	return psum;
}

func (gbLinear *GBLinear) PredictLeaf(feat util.FVec, ntree_limit int) ([]int, error) {
	return nil, fmt.Errorf("gblinear does not support predict leaf index")
}

func (gbLinear *GBLinear) Weight(fid, gid int) float32 {
	return gbLinear.weights[fid*gbLinear.mparam.num_output_group+gid];
}

func (gbLinear *GBLinear) Bias(gid int) float32 {
	return gbLinear.weights[gbLinear.mparam.num_feature*gbLinear.mparam.num_output_group+gid];
}

type GBLinearParam struct {
	num_feature      int
	num_output_group int
	reserved         []int
}

func newGBLinearParam(reader *util.ModelReader) (*GBLinearParam, error) {
	gbLinearParam := new(GBLinearParam)
	var err error
	gbLinearParam.num_feature, err = reader.ReadInt();
	if err != nil {
		return gbLinearParam, err
	}
	gbLinearParam.num_output_group, err = reader.ReadInt();
	if err != nil {
		return gbLinearParam, err
	}
	gbLinearParam.reserved, err = reader.ReadIntArray(32);
	if err != nil {
		return gbLinearParam, err
	}
	_, err = reader.ReadInt();
	return gbLinearParam, err
}
