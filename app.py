from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ["GET", "POST"])
def predict_data():
    if request.method=="GET":
        return render_template('home.html')
    else:
        pass


if __name__=="__main__":
    app.run(debug=True)