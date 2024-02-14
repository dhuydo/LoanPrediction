## Common functionalities used in entire project

import os,sys
from src.exceptions import CustomException

import dill


def save_obj(file_path, obj):
    try:
        path = os.path.dirname(file_path)
        os.makedirs(path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_obj(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)    