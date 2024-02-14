import os,sys
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConFig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConFig()
    
    def get_data_transformation_obj(self):
        '''
        This function return data transformation
        '''
        
        try:
            cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'Credit_History', 'Property_Area']
            num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term']
            logging.info(f'Numerical columns: {num_features}')
            logging.info(f'Categorical columns: {cat_features}')
            
            logging.info('Create transformer object')
            cat_pipe = Pipeline([
                ('ohe', OneHotEncoder(sparse_output=False)),
                ('scaler', StandardScaler())
            ])
            num_pipe = Pipeline([
                ('log', FunctionTransformer(lambda x: np.log(x+1), validate=True)),
                ('scaler', StandardScaler())
            ])
            pipe = ColumnTransformer([
                ('cat_pipe', cat_pipe, cat_features),
                ('num_pipe', num_pipe, num_features)
            ])
            logging.info('Data transformer was created')
            
            return pipe 
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, val_path):
        try:
            logging.info('Loading train, val data')
            train = pd.read_csv(train_path)
            val = pd.read_csv(val_path)
            
            X_train = train.drop('Loan_Status',axis=1)
            X_val = val.drop('Loan_Status', axis=1)
            
            y_train = train.Loan_Status
            y_val = val.Loan_Status
            
            logging.info('Get data transformer')
            data_transformer = self.get_data_transformation_obj()
            
            logging.info('Apply transformation')
            train_transform = data_transformer.fit_transform(X_train)
            val_transform = data_transformer.transform(X_val)
            
            df_train = np.c_[train_transform, np.array(y_train)]
            df_val = np.c_[val_transform, np.array(y_val)]
            logging.info('Data transformation completed')
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_file_path, 
                obj = data_transformer
            )
            
            return(
                df_train, df_val, self.data_transformation_config.preprocessor_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            
        
        
        