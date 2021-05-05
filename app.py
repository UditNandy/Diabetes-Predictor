import numpy as np
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

scaler = StandardScaler()


df=pd.read_csv('diabetes.csv')
df=df.drop(columns='Outcome')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    ls=[x for x in request.form.values()]

    df1= {'Pregnancies': ls[0], 'Glucose': ls[1], 'BloodPressure': ls[2], 'SkinThickness': ls[3], 'Insulin': ls[4], 'BMI': ls[5],
              'DiabetesPedigreeFunction': ls[6], 'Age': ls[7]}

    df1=df.append(df1, ignore_index=True)
    d = scaler.fit_transform(df1)
    l = d[d.shape[0] - 1, :]
    l1 = np.reshape(l,(1,8))
    df.drop(df.shape[0] - 1, inplace=True)
    my_prediction = model.predict(l1)
    if(my_prediction[0]==1):
        return render_template('index.html',prediction_text='You are having Diabetes')
    else:
        return render_template('index.html', prediction_text='You are not having diabetes')


if __name__ == "__main__":
    app.run(debug=True)
