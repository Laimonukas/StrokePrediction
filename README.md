# Stroke Prediction

## Introduction
This is a project to predict the probability of a patient getting a stroke based on the given features. The dataset used is from [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). 

Scikit-learn pipelines were used to preprocess the data and train the model. The model itself is an ensamble of different classifiers.

Main focus was to get a high recall score, as it is important to correctly identify patients that are at risk of getting a stroke.

## Summary
- Performed EDA on the dataset:
    - Overall:
        - dataset has missing values
        - some features might not be useful
        - some features have faulty data types
        - dataset is imbalanced
    - Checked for:
        - correlation between features:
            - Numerical features have low/no correlation
            - Categorical features have low/no association
            - Numerical and categorical features have some correlation (e.g. age and ever_married or stroke)
        - distribution of features
        - outliers
- Performed Statistical Inference:
    - Tested if the mean of numerical features is different for patients that had a stroke and patients that did not have a stroke. Found that there is statistically significant difference in the mean of age, avg_glucose_level, and bmi.
    - Tested if the proportion of patients that had a stroke is different for different categories of categorical features. Found that there is statistically significant difference in the proportion of patients that had a stroke for some of the categorical features (e.g. smoking_status).
- Machine Learning Model:
    - Handled data preprocessing,- one-hot encoding for categorical features, imputing missing values, scaling numerical features etc.
    - Feature engineering e.g. risk_score_all - a combination of age, hypertension, heart_disease, avg_glucose_level, and bmi
    - Handled feature selection
    - Created a pipeline to preprocess the data, engineer extra features and do feature selection
    - Did hyperparameter tuning for a selection of classifiers using created pipeline.
    - Created an Voting ensemble of classifiers using the best classifiers from hyperparameter tuning.

## Usage

Web app is available at: [Streamlit: Stroke Prediction](https://strokeprediction-en3nmnodfue5g2az3whhfj.streamlit.app/)

To run the project locally:

Notebooks:
- Stroke_prediction.ipynb: Main notebook with interactive plotly plots.
- Stroke_prediction_static.ipynb: Notebook with static (png) plots.

Streamlit:
- streamlit_app.py - Streamlit app to predict the probability of a patient getting a stroke.

`streamlit run streamlit_app.py`

## Prerequisites
- Python 3
- Jupyter Notebook - for notebooks
- Streamlit - for running the web app locally
- dataset: [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Requirements
`
kagglehub
polars
numpy
plotly
scipy
scikit-learn
imbalanced-learn
streamlit
dill
jupyter
`