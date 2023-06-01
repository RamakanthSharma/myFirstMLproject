from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipelines.predict_pipeline import customData, predictPipeline

application=Flask(__name__)

app = application

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ["GET", "POST"])
def predict_data():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data = customData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            writing_score = float(request.form.get("writing_score")),
            reading_score = float(request.form.get("reading_score"))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = predictPipeline()
        result = predict_pipeline.predict(pred_df)
        return render_template('home.html', result=result[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")
