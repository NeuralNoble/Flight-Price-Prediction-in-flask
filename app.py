import pandas as pd
import numpy as np
import joblib
from flask import Flask, url_for, render_template
from forms import InputForm
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_extract=None, format="mixed", prefix=None):
        self.features_to_extract = features_to_extract
        self.format = format
        self.prefix = prefix

    def __sklearn_tags__(self):
        return {
            'allow_nan': True,
            'requires_y': False,
            'requires_fit': False,
            'preserves_dtype': True,
            'X_types': ['2darray', 'string']
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()

        # Handle both date and time formats
        if 'hour' in self.features_to_extract or 'minute' in self.features_to_extract:
            # For time columns
            X_datetime = pd.to_datetime(X.iloc[:, 0].map(lambda x: f'2000-01-01 {x}'))
        else:
            # For date column
            X_datetime = pd.to_datetime(X.iloc[:, 0])

        features = pd.DataFrame()

        feature_extractors = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day_of_month': lambda x: x.dt.day,
            'day_of_week': lambda x: x.dt.dayofweek,
            'week': lambda x: x.dt.isocalendar().week,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute
        }

        for feature in self.features_to_extract:
            if feature in feature_extractors:
                features[feature] = feature_extractors[feature](X_datetime)

        return features.values

    def get_feature_names_out(self, input_features=None):
        if self.prefix:
            return [f"{self.prefix}_{feature}" for feature in self.features_to_extract]
        return [f"{feature}" for feature in self.features_to_extract]


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

# Now load the model
try:
    model = joblib.load("model.joblib")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        x_new = pd.DataFrame(dict(
            airline=[form.airline.data],
            date_of_journey=[form.date_of_journey.data.strftime("%Y-%m-%d")],
            source=[form.source.data],
            destination=[form.destination.data],
            dep_time=[form.dep_time.data.strftime("%H:%M:%S")],
            arrival_time=[form.arrival_time.data.strftime("%H:%M:%S")],
            duration=[form.duration.data],
            total_stops=[form.total_stops.data],
            additional_info=[form.additional_info.data]
        ))
        prediction = model.predict(x_new)[0]
        message = f"The predicted price is {prediction:,.0f} INR!"
    else:
        message = "Please provide valid input details!"
    return render_template("predict.html", title="Predict", form=form, output=message)


if __name__ == "__main__":
    app.run(debug=True)