from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application
# import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
Standard_Scaler = pickle.load(open('models/scaler.pkl', 'rb'))

print("Ridge model:", type(ridge_model))
print("Scaler:", type(Standard_Scaler))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':

        feature1 = float(request.form.get('feature1'))
        feature2 = float(request.form.get('feature2'))
        feature3 = float(request.form.get('feature3'))
        feature4 = float(request.form.get('feature4'))
        feature5 = float(request.form.get('feature5'))
        feature6 = float(request.form.get('feature6'))
        feature7 = float(request.form.get('feature7'))
        feature8 = float(request.form.get('feature8'))
        feature9 = float(request.form.get('feature9'))
        new_data_scaled=Standard_Scaler.transform([[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9]])
        result=ridge_model.predict(new_data_scaled)
        return render_template(
                    'home.html',results=result[0]
                    
                )
    

    else:
        return render_template('home.html')
if __name__ == "__main__" :
    app.run(host="0.0.0.0")

 