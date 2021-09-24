import optuna
import numpy as np
import pandas as pd
from functools import partial
from . import model_bank
import mlflow
from .AAMPreprocessor import AAMPreprocessor
import joblib
from .FastAIutils import *
from .metrics import model_metrics, pretty_scores, get_scores
from loguru import logger
from pathlib import Path
from tabulate import tabulate
import pprint

class ProjectConfigurator:
    def __init__(self, config) -> None:
        if config:
            self.key_attrs = [i for i in dir(config) if not i.startswith('__')]
            for key in self.key_attrs:
                setattr(self, key, getattr(config, key))
        self.create_project_folder()
        self.add_logger_path()
        self.add_project_config_to_logs(config)

    def create_project_folder(self):
        self.output_path = Path(self.BASE_OUTPUT_PATH) / Path(self.PROJECT_NAME) / Path(self.SUB_PROJECT_NAME)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.output_path / 'models'
        self.models_path.mkdir(parents=True, exist_ok=True)
    # def copy_config_file(self):
    #     import shutil
    #     shutil.copy('config.py', str(self.output_path))

    def add_logger_path(self):
        self.logger = logger
        self.logger.add(str(self.output_path/'logfile.log'))
    
    def add_project_config_to_logs(self, config):
        bc_attrs = {i : getattr(config, i) for i in self.key_attrs }
        self.logger.info('\n'+pprint.pformat(bc_attrs))

class ARKAutoML(AAMPreprocessor):
    def __init__(self, data = None, config=None,
                n_folds= 5, eval_metric='recall', 
                n_trials=10, model_algos=['xgb','rf'], loading=False):
        self.config = ProjectConfigurator(config)
        if not loading:
            super().__init__(data, cat_cols=config.cat_cols, cont_cols=config.cont_cols, y_names=config.TARGET_COL, 
                            n_folds=n_folds, fold_method=config.FOLD_METHOD)

            self.eval_metric = eval_metric
            self.n_trials = n_trials
            self.model_algos = model_algos
            self.logger = self.config.logger
            self.total_features = len(self.cat_cols + self.cont_cols)
            self.mpb = model_bank.ModelParamBank(total_features = self.total_features)

    def create_optuna_optimization(self):
        self.study = optuna.create_study(direction='maximize', study_name=self.config.PROJECT_NAME, load_if_exists=True)
        mlflow.set_experiment(self.config.PROJECT_NAME)
        optimization_function = partial(self.objective)
        self.study.optimize(optimization_function, n_trials=self.n_trials)

    def objective(self, trial):
        valid_metrics = {}

        for fold in range(self.n_folds):
            self.mpb = model_bank.ModelParamBank(total_features = self.total_features, trial = trial)
            # self.trial_number_model[trial.number] = model_algo
            model_algo = trial.suggest_categorical("model_algo", self.model_algos)
            model = self.mpb.get_model_with_optuna_params(model_algo)
            model.fit(self.X_train[fold], self.y_train[fold])
            # train_metrics = self.model_metrics(model, self.X_train[fold], self.y_train[fold])
            valid_metrics[fold] = model_metrics(model, self.X_test[fold], self.y_test[fold], self.logger)
        cross_validated_metrics = pd.DataFrame(valid_metrics).mean(axis=1).to_dict()
        self.logger.info(f'''Trial No : {trial.number}, {self.eval_metric} : {np.round(cross_validated_metrics[self.eval_metric], 4)}, Params : {trial.params}
                        {pretty_scores(cross_validated_metrics)}''')
        
        with mlflow.start_run():
            mlflow.log_params(trial.params)
            for fold in range(self.n_folds): mlflow.log_metrics(valid_metrics[fold]) # metrics for each fold
            mlflow.log_metrics(cross_validated_metrics) # Adding the cross validated metrics

            tags = {
                        'eval_metric' : self.eval_metric,
                        'model_type'  : 'classification',
                        'model_algo'  : model_algo,
                        'train_shape' : self.X_train[fold].shape,
                        'test_shape' : self.X_test[fold].shape,
                        'sub_project' : self.config.SUB_PROJECT_NAME
                    }
            mlflow.set_tags(tags)
        return cross_validated_metrics[self.eval_metric]

    @staticmethod
    def calculate_metrics_based_on_different_cut_offs(model, X, y, cut_offs):
        full_metrics = []
        for co in cut_offs:
            loop_dict = get_scores(y, np.where(model.predict_proba(X)[:,1]>co, 1, 0))
            loop_dict['prob_cut_off'] = co
            full_metrics.append(loop_dict)
        cols = ['prob_cut_off'] + [i for i in loop_dict.keys() if i!='prob_cut_off'] #Reordering the columns
        return pd.DataFrame(full_metrics)[cols]

    def get_feature_importance(self, model, model_algo):
        if model_algo == 'xgb':
            fi = model.get_booster().get_fscore()
            fi_df = pd.DataFrame(fi, index=['importance']).T
            return (
                        (fi_df/fi_df['importance'].sum()).reset_index()
                            .sort_values('importance', ascending=False)
                            .rename(columns={'index':'features'})
                    )
        elif model_algo in ('lgb', 'rf'):
            fi = (
                    pd.DataFrame(model.feature_importances_, self.X.columns, columns=['importance'])
                    .sort_values('importance', ascending=False)
                    .reset_index()
                    .rename(columns={'index':'features'})
                    )
            fi['importance'] = fi['importance'] / fi['importance'].sum()
            return fi

    def get_crossvalidated_results_for_best_model(self, folds, params_dict = None, model_algo = None,
                                                         cut_offs=[0.5,.55,.6,.65,.7]):
        if pd.isnull(params_dict):
            params_dict = dict(self.study.best_trial.params.items())
            model_algo = params_dict.pop('model_algo')
        model = self.mpb.get_fresh_model(model_name = model_algo, params=params_dict)
        self.logger.info(f'Best Model : {model_algo}, Model Params : {params_dict}')
        valid_metrics = {}
        metrics_by_cut_off = []
        feature_importances = []
        for fold in range(folds):
            model.fit(self.X_train[fold], self.y_train[fold])
            self.logger.info(f'Fold : {fold+1}, Train Shape : {self.X_train[fold].shape}, Test Shape : {self.X_test[fold].shape}')
            valid_metrics[fold] = model_metrics(model, self.X_test[fold], self.y_test[fold], self.logger, print_metrics=True)
            
            fold_metrics_by_cut_off =  self.calculate_metrics_based_on_different_cut_offs(model, self.X_test[fold], self.y_test[fold], cut_offs)
            fold_metrics_by_cut_off['fold'] = fold
            metrics_by_cut_off.append(fold_metrics_by_cut_off)
            
            feature_importances.append(self.get_feature_importance(model, model_algo))
        
        full_fi = pd.concat(feature_importances)
        full_fi = (full_fi.groupby('features')['importance'].sum() / full_fi['importance'].sum()).reset_index().sort_values('importance', ascending=False).reset_index(drop=True)
        self.logger.info(f"Feature Importance : \n{tabulate(full_fi.head(30), headers='keys', tablefmt='psql')}")

        metrics_by_cut_off = pd.concat(metrics_by_cut_off)
        final_metrics_by_cut_off = metrics_by_cut_off.drop('fold', axis=1).groupby('prob_cut_off').mean().reset_index()
        self.logger.info(f"Metrics by different Cut Offs : \n{tabulate(final_metrics_by_cut_off, headers='keys', tablefmt='psql', showindex=False)}")
        
        cross_validated_metrics = pd.DataFrame(valid_metrics).mean(axis=1).to_dict()
        self.logger.info(f'*** Cross Validated Final Results : *** \n{pretty_scores(cross_validated_metrics)}')

        with mlflow.start_run():
            mlflow.log_params(params_dict)
            for fold in range(folds): mlflow.log_metrics(valid_metrics[fold]) # metrics for each fold
            mlflow.log_metrics(cross_validated_metrics) # Adding the cross validated metrics

            tags = {
                        'eval_metric' : self.eval_metric,
                        'model_type'  : 'classification',
                        'model_algo'  : model_algo,
                        'train_shape' : self.X_train[fold].shape,
                        'test_shape' : self.X_test[fold].shape,
                        'sub_project' : self.config.SUB_PROJECT_NAME
                    }
            mlflow.set_tags(tags)
        return cross_validated_metrics, final_metrics_by_cut_off        

    def create_only_one_train_and_one_valid(self, valid):
        '''
        This method is NOT RECOMMENDED!!
        But can be used in low computational resource scenatio.
        '''
        self.X_train[0], self.y_train[0] = self.X, self.y
        self.create_new_oos_test(valid)
        self.X_test[0], self.y_test[0] = self.X_valid, self.y_valid
        self.n_folds = 1

    def train_best_model(self, params_dict = None, model_algo = None):
        # super().create_no_split_train()
        if pd.isnull(params_dict):
            params_dict = dict(self.study.best_trial.params.items())
            model_algo = params_dict.pop('model_algo')
            self.logger.info(f'Training "{model_algo}" Model with Best Params : {params_dict}')
        
        self.final_model = self.mpb.get_fresh_model(model_name = model_algo, params=params_dict)
        self.final_model.fit(self.X, self.y)
        model_metrics(self.final_model, self.X, self.y, self.logger, print_metrics=True)
        fi = self.get_feature_importance(self.final_model, model_algo)
        self.logger.info(f"Best Model Feature Importance : \n{tabulate(fi.head(30), headers='keys', tablefmt='psql')}")

    def save_best_model(self,
                        preprocessor_file_name = 'best_model_preprocessor.pkl',
                        model_filename = 'best_model.pkl'):
        full_preprocessor_file_name = self.config.models_path / preprocessor_file_name
        full_model_filename = self.config.models_path / model_filename
        self.preprocessor_final.export(full_preprocessor_file_name)
        joblib.dump(self.final_model, full_model_filename)
        self.logger.info(full_preprocessor_file_name)
        self.logger.info(full_model_filename)

    def load_best_model(self,
                        preprocessor_file_name = 'best_model_preprocessor.pkl',
                        model_filename = 'best_model.pkl'):
        full_preprocessor_file_name = self.config.models_path / preprocessor_file_name
        full_model_filename = self.config.models_path / model_filename
        self.preprocessor_final = load_pandas(full_preprocessor_file_name)
        self.final_model = joblib.load(full_model_filename)

    def score_validation_data(self, data,
                                    cut_offs=[0.5,.55,.6,.65,.7]):
        self.create_new_oos_test(data)
        self.logger.info(f'Scoring Validation Data : Validation Shape : {self.X_valid.shape}')
        scores = model_metrics(self.final_model, self.X_valid, self.y_valid, self.logger, print_metrics=True)
        metrics_cut_off = self.calculate_metrics_based_on_different_cut_offs(self.final_model, self.X_valid, self.y_valid, cut_offs)
        self.logger.info(f"Metrics by different Cut Offs : \n{tabulate(metrics_cut_off, headers='keys', tablefmt='psql', showindex=False)}")
        return scores, metrics_cut_off
    

    def predict(self, data):
        self.predict_data = self.preprocessor_final.train.new(data)
        self.predict_data.process()
        xcols = self.preprocessor_final.train.xs.columns
        return self.final_model.predict(self.predict_data[xcols]), self.final_model.predict_proba(self.predict_data[xcols])

if __name__=='__main__':
    pass