from .create_folds import create_folds
from .FastAIutils import *
import pandas as pd
class AAMPreprocessor:
    def __init__(self, data, cat_cols = None, cont_cols = None, y_names=None,
                n_folds= 5, fold_method = 'StratifiedKFold'):
        self.data = data.copy()
        self.cat_cols, self.cont_cols = cat_cols, cont_cols
        self.y_names = y_names
        self.n_folds = n_folds
        self.fold_method = fold_method
        self.X_train, self.y_train, self.X_test, self.y_test = {}, {}, {}, {}
        self.preprocessor = {}
        if n_folds:
            self.create_folds()
        self.create_no_split_train()
        
    def create_folds(self):
        self.data = create_folds(self.data, self.n_folds, self.y_names, self.fold_method)
        for fold in range(self.n_folds):
            self.create_train_test(fold)

    def create_train_test(self, fold):
        self.fold = fold
        self.train_data, self.test_data = self.data.query('kfold!=@fold'), self.data.query('kfold==@fold')
        self.create_train()
        self.create_test()

    def create_train(self):
        self.preprocessor[self.fold] = TabularPandas(self.train_data, procs=[Categorify, FillMissing(add_col=None), Normalize],
                                            cat_names=self.cat_cols,
                                            cont_names=self.cont_cols,
                                            y_names=self.y_names
                                            )
        self.X_train[self.fold], self.y_train[self.fold] = self.preprocessor[self.fold].train.xs, self.preprocessor[self.fold].train.ys.values.ravel()


    def create_test(self):
        self.oosd = self.preprocessor[self.fold].train.new(self.test_data)
        self.oosd.process()
        cols = self.preprocessor[self.fold].train.xs.columns
        y_col = self.preprocessor[self.fold].y_names[0]
        self.X_test[self.fold], self.y_test[self.fold] = self.oosd[cols], self.oosd[y_col]

    def create_no_split_train(self):
        self.preprocessor_final = TabularPandas(self.data, procs=[Categorify, FillMissing(add_col=None), Normalize],
                                            cat_names=self.cat_cols,
                                            cont_names=self.cont_cols,
                                            y_names=self.y_names
                                            )
        self.X, self.y = self.preprocessor_final.train.xs, self.preprocessor_final.train.ys.values.ravel()

    def create_new_oos_test(self, data):
        self.oos_valid = self.preprocessor_final.train.new(data)
        self.oos_valid.process()
        cols = self.preprocessor_final.train.xs.columns
        y_col = self.preprocessor_final.y_names[0]
        self.X_valid, self.y_valid = self.oos_valid[cols], self.oos_valid[y_col]

