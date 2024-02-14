import os,sys
from src.exceptions import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConFig, DataTransformation
from src.components.model_training import ModelTrainingConfig, ModelTraining
from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str = os.path.join('artifacts', 'val.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    
class Dataingestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info('Read dataset as dataframe')
            df = pd.read_csv('data/train_cleaned.csv')
            
            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)
            
            logging.info('Initiate train val split')
            train, val = train_test_split(df, stratify=df.Loan_Status,test_size=0.2, random_state=29)
            
            train.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            val.to_csv(self.ingestionConfig.val_data_path, index=False, header=True)
            
            logging.info('Data ingestion completed')
            
            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.val_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=='__main__':
    obj = Dataingestion()
    train_path, val_path = obj.initiate_data_ingestion()
   
    data_tranformation = DataTransformation()
    df_train, df_val, _ = data_tranformation.initiate_data_transformation(train_path, val_path)
    model_training = ModelTraining()
    print(model_training.initiate_model_training(df_train, df_val))
    

