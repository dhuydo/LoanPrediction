from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from src.pipelines.predict_pipeline import GetData, PredictPipeline

app = Flask(__name__)

## Route

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        input = [instance for instance in request.form.values()]
        data = GetData(input).get_data_asframe()
        y_pred = PredictPipeline().predict(data)[:,1]
        result = f'Loan Application approved: {(y_pred>=0.5)[0]}, with probability of being approved: {(y_pred)[0]:.4f}'
        return render_template('home.html',result=result)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=1001, debug=True)