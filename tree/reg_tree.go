package tree

import "xgboost4go-predictor/util"

type RegTree struct {
	param *Param
	nodes []*Node
	stats []*RTreeNodeStat
}

func (rt *RegTree) GetLeafIndex(feat util.FVec, root_id int) int {
	pid := root_id
	n := rt.nodes[pid]
	for pid = root_id; !n._isLeaf; pid = n.next(feat) {
		n = rt.nodes[pid]
	}

	return pid;
}

func (rt *RegTree) GetLeafValueFloat(feat util.FVec, root_id int) float32 {
	n := rt.nodes[root_id]
	for ; !n._isLeaf; n = rt.nodes[n.next(feat)] {
		n = rt.nodes[root_id]
	}

	return n.leaf_value;
}

type Param struct {
	num_roots        int
	num_nodes        int
	num_deleted      int
	max_depth        int
	num_feature      int
	size_leaf_vector int
	reserved         []int
}

type Node struct {
	parent_      int
	cleft_       int
	cright_      int
	sindex_      int
	leaf_value   float32
	split_cond   float32
	_defaultNext int
	_splitIndex  int
	_isLeaf      bool
}

type RTreeNodeStat struct {
	Loss_chg       float32
	Sum_hess       float32
	Base_weight    float32
	Leaf_child_cnt int
}

func (rt *RegTree) LoadModel(reader *util.ModelReader) error {
	param, err := newParam(reader)
	if err != nil {
		return err
	}
	rt.param = param
	rt.nodes = make([]*Node, param.num_nodes)

	for i := 0; i < param.num_nodes; i++ {
		rt.nodes[i], err = newNode(reader);
		if err != nil {
			return err
		}
	}

	rt.stats = make([]*RTreeNodeStat, param.num_nodes)

	for i := 0; i < param.num_nodes; i++ {
		rt.stats[i], err = newRTreeNodeStat(reader);
		if err != nil {
			return err
		}
	}
	return err
}

func newParam(reader *util.ModelReader) (*Param, error) {
	param := new(Param)
	var err error
	param.num_roots, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.num_nodes, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.num_deleted, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.max_depth, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.num_feature, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.size_leaf_vector, err = reader.ReadInt();
	if err != nil {
		return param, err
	}
	param.reserved, err = reader.ReadIntArray(31);
	return param, err
}

func newNode(reader *util.ModelReader) (*Node, error) {
	node := new(Node)
	var err error
	node.parent_, err = reader.ReadInt();
	if err != nil {
		return node, err
	}
	node.cleft_, err = reader.ReadInt();
	if err != nil {
		return node, err
	}
	node.cright_, err = reader.ReadInt();
	if err != nil {
		return node, err
	}
	node.sindex_, err = reader.ReadInt();
	if err != nil {
		return node, err
	}
	if (node.is_leaf()) {
		node.leaf_value, err = reader.ReadFloat()
		node.split_cond = 0.0 / 0.0;
	} else {
		node.split_cond, err = reader.ReadFloat()
		node.leaf_value = 0.0 / 0.0;
	}

	node._defaultNext = node.cdefault();
	node._splitIndex = node.split_index();
	node._isLeaf = node.is_leaf();
	return node, nil
}

func newRTreeNodeStat(reader *util.ModelReader) (*RTreeNodeStat, error) {
	rTreeNodeStat := new(RTreeNodeStat)
	var err error
	rTreeNodeStat.Loss_chg, err = reader.ReadFloat();
	if err != nil {
		return rTreeNodeStat, err
	}
	rTreeNodeStat.Sum_hess, err = reader.ReadFloat();
	if err != nil {
		return rTreeNodeStat, err
	}
	rTreeNodeStat.Base_weight, err = reader.ReadFloat();
	if err != nil {
		return rTreeNodeStat, err
	}
	rTreeNodeStat.Leaf_child_cnt, err = reader.ReadInt();
	return rTreeNodeStat, err
}

func (n *Node) is_leaf() bool {
	return n.cleft_ == -1
}

func (n *Node) split_index() int {
	return int(int64(n.sindex_) & 2147483647)
}

func (n *Node) cdefault() int {
	if n.default_left() {
		return n.cleft_
	} else {
		return n.cright_
	}
}

func (n *Node) default_left() bool {
	return n.sindex_>>31 != 0;
}

func (n *Node) next(feat util.FVec) int {
	fvalue := feat.Fvalue(n._splitIndex);
	if (fvalue != fvalue) {
		return n._defaultNext;
	} else {
		if fvalue < n.split_cond {
			return n.cleft_
		} else {
			return n.cright_
		}
	}
}
