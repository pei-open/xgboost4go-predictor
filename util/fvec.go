package util

type FVec interface {
	Fvalue(index int) float32
}

type FVecFloat32ArrayImpl struct {
	Values         []float32;
	TreatsZeroAsNA bool;
}

func (f FVecFloat32ArrayImpl) Fvalue(index int) float32 {
	if (len(f.Values) <= index) {
		return 0.0 / 0.0;
	} else {
		result := f.Values[index]
		if f.TreatsZeroAsNA && result == 0.0 {
			return 0.0 / 0.0
		} else {
			return result
		}
	}
}

type FVecFloat64ArrayImpl struct {
	Values         []float64;
	TreatsZeroAsNA bool;
}

func (f FVecFloat64ArrayImpl) Fvalue(index int) float32 {
	if (len(f.Values) <= index) {
		return 0.0 / 0.0;
	} else {
		result := f.Values[index]
		if f.TreatsZeroAsNA && result == 0.0 {
			return 0.0 / 0.0
		} else {
			return float32(result)
		}
	}
}

type FVecMapFloat32Impl struct {
	Values map[int]float32
}

func (f FVecMapFloat32Impl) Fvalue(index int) float32 {
	value, ok := f.Values[index]
	if ok {
		return 0.0 / 0.0
	} else {
		return value
	}
}

type FVecMapFloat64Impl struct {
	Values map[int]float64
}

func (f FVecMapFloat64Impl) Fvalue(index int) float32 {
	value, ok := f.Values[index]
	if ok {
		return 0.0 / 0.0
	} else {
		return float32(value)
	}
}

func FromArrayFloat32(values []float32, treatsZeroAsNA bool) FVec {
	return FVecFloat32ArrayImpl{values, treatsZeroAsNA}
}

func FromArrayFloat64(values []float64, treatsZeroAsNA bool) FVec {
	return FVecFloat64ArrayImpl{values, treatsZeroAsNA}
}

func FromMapFloat32(values map[int]float32) FVec {
	return FVecMapFloat32Impl{values}
}

func FromMapFloat64(values map[int]float64, treatsZeroAsNA bool) FVec {
	return FVecMapFloat64Impl{values}
}
