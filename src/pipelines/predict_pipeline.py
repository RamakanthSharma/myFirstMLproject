import sys
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object


class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        data_scaled = preprocessor.transform(features)
        prediction = model.predict(data_scaled)
        return prediction

class customData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str,
                 lunch: str, test_preparation_course: str, writing_score: float, reading_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score" : [self.writing_score],
                "reading_score" : [self.reading_score]
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e,sys)