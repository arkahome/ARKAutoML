from sklearn import ensemble,linear_model
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC

class ModelParamBank:
    def __init__(self, total_features, trial=None) -> None:
        self.trial = trial
        self.total_features = total_features
        self.initiate_gs_params()
        self.initiate_models()
                                
    def get_gridsearch_params(self, model_name):
        return self.gridsearch_params[model_name]['gridsearch']
    
    def get_optuna_params(self, model_name):
        trial = self.trial
        if model_name == 'rf':
            self.optuna_params = {
                                        "max_depth": trial.suggest_int("max_depth", 2, self.total_features),
                                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
                                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
                                    }

        elif model_name == 'xgb':
            self.optuna_params = {
                                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                                        'eta': trial.suggest_uniform('eta', 0.01, .1),
                                        'max_depth' : trial.suggest_int('max_depth', 2, self.total_features),
                                        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 7),
                                        "subsample" : trial.suggest_discrete_uniform('subsample', 0.6, 1, 0.1),
                                        "colsample_bytree" : trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1, 0.1),
                                        "lambda" : trial.suggest_uniform('lambda', 0.01, 1),
                                        "alpha" : trial.suggest_uniform('alpha', 0, 1)
                                    }
        elif model_name == 'lgb':
            self.optuna_params = {
                                        'num_leaves' : trial.suggest_int('num_leaves', 6, 50),
                                        'min_child_samples' : trial.suggest_int('min_child_samples', 100, 500),
                                        'min_child_weight' : trial.suggest_int("min_child_weight", 1, 7),
                                        'subsample' : trial.suggest_discrete_uniform('subsample', 0.6, 1, 0.1),
                                        'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1, 0.1),
                                        'reg_alpha' : trial.suggest_discrete_uniform('reg_alpha', 0.1, 100, 1),
                                        'reg_lambda' : trial.suggest_discrete_uniform('reg_lambda', 0.1, 100, 1),
            }
        elif model_name == 'svm' : 
            self.optuna_params = {
                                        'C' : trial.suggest_uniform('C',.001, 1000),
                                        'gamma' : trial.suggest_categorical('gamma', ['auto']),
                                        'class_weight' : trial.suggest_categorical('class_weight', ['balanced', None])
                                    }
        return self.optuna_params

    def get_model_with_optuna_params(self, model_name, model_type='classification'):
        params = self.get_optuna_params(model_name)
        if model_name in ['svm']:
            return self.models[model_type][model_name](**params)
        return self.models[model_type][model_name](**params, n_jobs=-1)

    def get_fresh_model(self, model_name, params, model_type='classification'):
        if model_name in ['svm']:
            return self.models[model_type][model_name](**params)
        return self.models[model_type][model_name](**params, n_jobs=-1)

    def initiate_models(self):
        self.models = {
                            'regression':{
                                'rf':ensemble.RandomForestRegressor,
                                'xgb':xgb.XGBRFRegressor
                            },
                            'classification':{
                                'logreg' : linear_model.LogisticRegression,
                                'rf':ensemble.RandomForestClassifier,
                                'xgb':xgb.XGBClassifier,
                                'lgb' : lgb.LGBMClassifier,
                                'svm' : SVC,

                            }
                        }
    def initiate_gs_params(self):

        self.gridsearch_params = {
                                        'rf' : {
                                            'gridsearch' : {
                                                'n_estimators' : [120, 300, 500, 800, 1200],
                                                'max_depth' : [5, 8, 15, 25, 30, 40, None],
                                                'max_samples_split' : [1, 2, 5, 10, 15, 100],
                                                'max_features' : ['log2', 'sqrt', None]
                                            },

                                        'xgb' : {
                                            'gridsearch' : {
                                                "eta" : [0.01, 0.015, 0.025, 0.05, 0.1],
                                                "gamma" : [0.05, .1, .3, .5, .7, .9, 1], #0.05 to .1
                                                "max_depth" : [3, 5, 7, 9, 12, 15, 17, 25],
                                                "min_child_weight" : [1, 3, 5, 7],
                                                "subsample" : [0.6, 0.7, .8, .9, 1],
                                                "colsample_bytree" : [.6, .7, .8, .9, 1],
                                                "lambda" : [0.01, 0.1 ], # 0.01 to 0.1 
                                                "alpha" : [0, .1, 0.5, 1]
                                            }
                                        },
                                        'svm' : {
                                            'gridsearch' : {
                                                'C' : [.001, .01, .01, 1, 10, 100, 1000], # Entire range
                                                'gamma' : ['auto'], # Tune by hand random search
                                                'class_weight' : ['balanced', None]
                                            },

                                        },
                                        'logisticregression'  : {
                                            'gridsearch' : {
                                                'penalty' : ['l1','l2'],
                                                'C' : [0.001, .01, 1, 10, 100]
                                            }
                                        },
                                        'lasso' : {
                                            'gridsearch' : {
                                                'alpha' : [0.1, 1, 10],
                                                'normalize' : [True, False]
                                            }
                                        },
                                        'k-neighbors' : {
                                            'gridsearch' : {
                                                'n_neighbors' : [2, 4, 8, 16,], #..
                                                'p' : [2, 3]
                                            }
                                        },
                                        'linearregression' : {
                                            'gridsearch' : {
                                                'fit_intercept' : [True, False],
                                                'normalize' : [True, False]
                                            }
                                        }
                                        }
                                    }