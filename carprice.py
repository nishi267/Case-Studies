
from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
import csv 
import json 
from flasgger import Swagger
import os

app=Flask(__name__)
Swagger(app)

pred1, train_X, train_y =pickle.load(open(r'D:\code-replicate\SVC1\svcmodel.pkl','rb'))


def functi(df):
    y = pd.get_dummies(df[['CarName','enginelocation','fueltype','aspiration','doornumber','carbody','drivewheel',
                         'enginetype','cylindernumber','fuelsystem']])
    x=df.drop(['CarName','enginelocation','fueltype','aspiration','doornumber','carbody','drivewheel','enginetype','cylindernumber','fuelsystem'], axis=1)
    z = x.join(y)
    X=z.drop(['price'], axis=1)
    return X

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    data = json.load(df_test) 
    mast=[]
    for i in data['Details']: 
        mast.append(i)
    df = pd.DataFrame(mast)
    df=functi(df)
    train_X['train']=1
    df['train']=0
    comb=pd.concat([train_X,df])
    predtest=comb[comb['train']==0]
    predtest.drop(['train'], axis=1, inplace=True)
    predtest = predtest.fillna(0)
    prediction = pred1.predict(predtest)

    #print(df_test.head())
    #prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    
    