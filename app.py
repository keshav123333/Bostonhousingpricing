import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import sklearn


app=Flask(__name__)

# Load the model
model=pickle.load(open("regmodel.pkl","rb"))
scalar=pickle.load(open("scaler.pkl","rb"))

#now we will create a home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # Get the data from the request and we will get the data from json format 
    data=request.json['data']
    new_data= caler.transform(np.array(list(data.values()).reshape(1,-1)))
    # Predict the result
    output=model.predict(new_data)
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)
# The above code is a Flask application that serves a machine learning model for predictions.
# It loads a pre-trained model and scaler from pickle files, defines a home route that renders an HTML template,
# and a prediction route that accepts JSON data, transforms it using the scaler, and returns the model's prediction as JSON.
# The application runs in debug mode, allowing for easy testing and development.


