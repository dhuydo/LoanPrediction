import os, sys
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_obj

import pandas as pd

class GetData:
    def __init__(self, input):
        self.input = input
    def get_data_asframe(self):
        columns_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'Credit_History', 'Property_Area', 'ApplicantIncome', 'CoapplicantIncome', 
            'LoanAmount', 'Loan_Amount_Term']
        df = pd.DataFrame(dict(zip(columns_names, self.input)), index=[0])
        logging.info('Import data as dataframe')
        
        return df
          
            
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, data):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')
        
        preprocessor = load_obj(preprocessor_path)
        model = load_obj(model_path)
        
        tranformed_data = preprocessor.transform(data)
        y_pred = model.predict_proba(tranformed_data)
        
        return y_pred
        
    