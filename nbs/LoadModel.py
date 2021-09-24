from fastai.tabular.all import *
import joblib

class LoadModel:
    def __init__(self, preprocessor_file_path,
                    model_path) : 
        self.load_best_model(preprocessor_file_path, model_path)
    def load_best_model(self,
                    preprocessor_file_path,
                    model_path):
        self.preprocessor_final = self.load_pandas(preprocessor_file_path)
        print('Model Features : \n', list(self.preprocessor_final.train.xs.columns))
        self.final_model = joblib.load(model_path)
    def predict(self, data):
        self.predict_data = self.preprocessor_final.train.new(data)
        self.predict_data.process()
        xcols = self.preprocessor_final.train.xs.columns
        return self.final_model.predict(self.predict_data[xcols]), self.final_model.predict_proba(self.predict_data[xcols])
    
    @staticmethod
    def load_pandas(fname):
        "Load in a `TabularPandas` object from `fname`"
        distrib_barrier()
        res = pickle.load(open(fname, 'rb'))
        return res