package gbm

import (
	"xgboost4go-predictor/tree"
	"xgboost4go-predictor/util"
	"xgboost4go-predictor/math"
)

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

func (gbTree *GBTree) PredictArray(values []float32, treatsZeroAsNA bool, ntree_limit int) []float32 {
	preds := make([]float32, gbTree.mparam.num_output_group)
	for gid := 0; gid < gbTree.mparam.num_output_group; gid++ {
		preds[gid] = gbTree.PredArray(values, treatsZeroAsNA)
	}

	return preds
}

func (gbTree *GBTree) PredictMap(values map[int]float32, ntree_limit int) []float32 {
	preds := make([]float32, gbTree.mparam.num_output_group)
	for gid := 0; gid < gbTree.mparam.num_output_group; gid++ {
		preds[gid] = gbTree.PredMap(values, gid, 0, ntree_limit)
	}

	return preds
}

func (gbTree *GBTree) PredictSingleFromArray(values []float32, treatsZeroAsNA bool) float32 {
	return gbTree.PredArray(values, treatsZeroAsNA)
}

func (gbTree *GBTree) PredictSingleFromMap(values map[int]float32) float32 {
	if (gbTree.mparam.num_output_group != 1) {
		return math.NAN
	} else {
		return gbTree.PredMap(values, 0, 0, 0)
	}
}

func (gbTree *GBTree) PredMap(values map[int]float32, bst_group, root_index, ntree_limit int) float32 {
	trees := gbTree._groupTrees[bst_group]
	treeleft := ntree_limit
	if ntree_limit == 0 {
		treeleft = len(trees)
	}
	psum := FLOAT_32_0
	for i := 0; i < treeleft; i++ {
		psum += trees[i].GetLeafByMap(values, root_index)
	}

	return psum
}

func (gbTree *GBTree) PredArray(values []float32, treatsZeroAsNA bool) float32 {
	trees := gbTree._groupTrees[0]
	treeleft := len(trees)
	psum := FLOAT_32_0
	for i := 0; i < treeleft; i++ {
		psum += trees[i].GetLeafByArray(values, treatsZeroAsNA)
	}

	return psum
}

func (gbTree *GBTree) PredictLeafFromArray(values []float32, treatsZeroAsNA bool, root_index, ntree_limit int) []int {
	return gbTree.PredPathArray(values, treatsZeroAsNA, 0, ntree_limit)
}

func (gbTree *GBTree) PredictLeafFromMap(values map[int]float32, ntree_limit int) []int {
	return gbTree.PredPathMap(values, 0, ntree_limit)
}

func (gbTree *GBTree) PredPathArray(values []float32, treatsZeroAsNA bool, root_index, ntree_limit int) []int {
	var treeleft int
	if ntree_limit == 0 {
		treeleft = len(gbTree.trees)
	} else {
		treeleft = ntree_limit
	}
	leafIndex := make([]int, treeleft)
	for i := 0; i < treeleft; i++ {
		leafIndex[i] = gbTree.trees[i].GetLeafIndexByArray(values, treatsZeroAsNA, root_index)
	}

	return leafIndex
}

func (gbTree *GBTree) PredPathMap(values map[int]float32, root_index, ntree_limit int) []int {
	var treeleft int
	if ntree_limit == 0 {
		treeleft = len(gbTree.trees)
	} else {
		treeleft = ntree_limit
	}
	leafIndex := make([]int, treeleft)
	for i := 0; i < treeleft; i++ {
		leafIndex[i] = gbTree.trees[i].GetLeafIndexByMap(values, root_index)
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
