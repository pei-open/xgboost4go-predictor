package config

import "xgboost4go-predictor/learner"

var DEFAULT = new(Configuration)

type Configuration struct {
	ObjFunction learner.ObjFunction
}
