from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
pickle_in=open("classifier.pkl","rb")
classifer=pickle.load(pickle_in)
@app.route("/")
def welcome():
    return render_template("index2.html")
@app.route("/submit",methods=["GET","POST"])
def predict_bank_note():
    if request.method=="POST":
        variance=request.form["v"]
        skewness=request.form["sk"]
        curtosis=request.form["cu"]
        entropy=request.form["en"]
        
    prediction=classifer.predict([[variance,skewness,curtosis,entropy]])
    return render_template("index1.html",prediction=prediction)










if __name__=="__main__":
    app.run(debug=True)


