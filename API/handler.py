from flask import Flask, request, Response
from rossman.Rossman import Rossman
import pandas as pd
import pickle

# Loading Model
model = pickle.load(open('/home/jordanmalheiros/Estudismo/ds_producao/Rossman/model/model_rossman.pkl','rb'))

# Initialize API
app = Flask(__name__)

@app.route('/rossman/predict',methods=['POST'])

def rossman_predict():
    test_json = request.get_json()
    
    #There is json?!
    if test_json:
        if isinstance(test_json,dict): # Only one example
            test_raw = pd.DataFrame(test_json,index=[0])
        else: # Multiple Exames
            test_raw = pd.DataFrame(test_json,columns=test_json[0].keys())
            
        # Instance Rossman Class
        pipeline = Rossman()
        
        # Data Cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)
        
        # Data Preparation
        df3 = pipeline.data_preparation(df2)
        
        # Prediction
        df_response = pipeline.get_prediction(model,test_raw,df3)
        
        return df_response
        
        
    else:
        return Response('{}',status=200,mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')