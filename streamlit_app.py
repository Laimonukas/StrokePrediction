# Run: streamlit run streamlit_app.py
import streamlit as st

# other imports
from custom_modules.custom_pickler import DillWrapper
from custom_modules.constants import DISCRETE_COLOR_MAP, BASE_THEME, RNG
from custom_modules.custom_functions import (
    plot_numeric_scatter,
    plot_violin,
    plot_points_subplots,
)
from custom_modules.custom_functions import combine_data
from custom_modules.custom_functions import (
    plot_correlation_heatmap,
    continous_continous_correlation_matrix,
    categorical_categorical_correlation_matrix,
    plot_continous_categorical_correlation_matrix,
)

import polars as pl
import numpy as np

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

# env setup
st.set_page_config(layout="wide", page_title="Stroke Prediction", page_icon="🧠")
pio.templates.default = BASE_THEME
px.defaults.template = BASE_THEME
pio.templates[BASE_THEME].layout.colorway = DISCRETE_COLOR_MAP
pio.renderers.default = "notebook_connected"


# Basic function
@st.cache_data
def read_data():
    return pl.read_csv("datasets/healthcare-dataset-stroke-data.csv")


@st.cache_data
def load_model_obj():
    return DillWrapper().load_by_hash("5edda379ff46f6d1cdffde234e0bfa6a")


@st.cache_data
def fix_base_data(_data):
    X_train, X_test, y_train, y_test = train_test_split(
        _data.drop(["stroke", "id"]),
        _data["stroke"],
        test_size=0.2,
        random_state=RNG,
        stratify=_data["stroke"],
    )
    _data = combine_data(X_train, y_train, ["hypertension", "heart_disease", "stroke"])
    mean_bmi = _data.filter(pl.col("bmi") != "N/A")["bmi"].cast(pl.Float32).mean()
    filled_bmi = _data.select(pl.col("bmi").replace({"N/A": str(mean_bmi)})).cast(
        pl.Float32
    )["bmi"]
    _data = _data.select(pl.all().exclude("bmi")).with_columns(filled_bmi)
    return _data, X_train, y_train


# Streamlit code
model_obj = load_model_obj()
df = read_data()
df, X_train, y_train = fix_base_data(df)


selection = st.sidebar.selectbox("Select page:", ("Introduction", "EDA", "Prediction"))

match selection:
    case "Introduction":
        st.header("Introduction")
        st.write("This web app is a demo of a stroke prediction model.")
        st.write(
            "The model was trained on the dataset available [here](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)."
        )
        st.write(
            "The focus was to minimize the number of False Negatives, i.e., to minimize the number of people who are predicted to not have a stroke but actually do. "
            "As a result, the model has a high recall score, but a lower precision score. And overall mediocre accuracy."
        )
        st.write(
            "The model is a Voting Classifier, which combines the predictions of a BalancedBaggingClassifier(SVC), "
            "BalancedBaggingClassifier(PassiveAgressiveClassifier), BalancedBaggingClassifier(DecisionTree)"
        )
    case "EDA":
        st.header("Exploratory Data Analysis")
        eda_tab_scatter, eda_tab_corr, eda_tab_violin, eda_tab_point = st.tabs(
            ["Scatter Plot", "Correlation Matrix", "Violin Plot", "Point Plot"]
        )

        with eda_tab_scatter:
            st.subheader("Scatter Plot")

            target_col = st.selectbox("Select target column:", df.columns)
            columns = df.columns
            columns.remove(target_col)

            x_axis = st.selectbox("Select x-axis:", columns)
            y_columns = columns
            y_columns.remove(x_axis)
            y_axis = st.selectbox("Select y-axis:", y_columns)
            st.plotly_chart(
                plot_numeric_scatter(
                    data=df, columns=(x_axis, y_axis), target_col=target_col
                )
            )

        with eda_tab_corr:
            st.subheader("Correlation Matrix")
            corr_selection = st.selectbox(
                "Select correlation type:",
                (
                    "Continous-Continous",
                    "Categorical-Categorical",
                    "Continous-Categorical",
                ),
            )
            match corr_selection:
                case "Continous-Continous":
                    st.plotly_chart(
                        plot_correlation_heatmap(
                            continous_continous_correlation_matrix(
                                df["age", "avg_glucose_level", "bmi"]
                            )
                        )
                    )
                case "Categorical-Categorical":
                    st.plotly_chart(
                        plot_correlation_heatmap(
                            data=categorical_categorical_correlation_matrix(
                                df.select(pl.col(pl.String))
                            ),
                            title="Categorical-Categorical Correlation Matrix",
                            color_scale="Blues",
                            zmin=0,
                            sub_title="Cramer's V",
                        )
                    )
                case "Continous-Categorical":
                    st.plotly_chart(
                        plot_continous_categorical_correlation_matrix(
                            data=df,
                            categorical_columns=df.select(pl.col(pl.String)).columns,
                            continous_columns=df.select(
                                pl.all().exclude(pl.String)
                            ).columns,
                            title="Correlation Matrix",
                        )
                    )
            with eda_tab_violin:
                st.subheader("Violin Plot")
                target_col = st.selectbox(
                    "Select target column:",
                    ["ever_married", "hypertension", "heart_disease", "stroke"],
                    key="vp_target",
                )
                columns = df.select(pl.col(pl.String)).columns
                columns.remove(target_col)
                x_axis = st.selectbox("Select x-axis:", columns, key="vp_x")
                columns = df.select(pl.all().exclude(pl.String)).columns
                y_axis = st.selectbox("Select y-axis:", columns, key="vp_y")
                st.plotly_chart(
                    plot_violin(df, target_column=target_col, columns=[x_axis, y_axis])
                )
            with eda_tab_point:
                st.subheader("Point Plot")
                columns = df.select(pl.col(pl.String)).columns
                columns.remove("stroke")
                point_selection = st.selectbox(
                    "Select point plot type:",
                    (columns),
                )
                confidence = st.slider("Confidence Interval:", 0.1, 0.99, 0.95)
                st.plotly_chart(
                    plot_points_subplots(
                        df,
                        "stroke",
                        "Yes",
                        confidence,
                        [
                            point_selection,
                        ],
                    )
                )
    case "Prediction":
        st.header("Prediction")
        st.write(
            "This section allows you to predict if a person will have a stroke or not."
        )
        st.write("Please input the required data below:")
        input_data = {}
        data = df.to_pandas()
        input_data["gender"] = st.selectbox("Gender", data["gender"].unique())

        input_data["age"] = st.slider(
            "Age",
            min_value=int(1),
            max_value=int(data["age"].max()),
            value=50,
        )

        input_data["hypertension"] = st.selectbox("Hypertension", [0, 1])

        input_data["heart_disease"] = st.selectbox("Heart Disease", [0, 1])

        input_data["ever_married"] = st.selectbox("Ever Married", ["Yes", "No"])

        input_data["work_type"] = st.selectbox("Work Type", data["work_type"].unique())

        input_data["Residence_type"] = st.selectbox(
            "Residence Type", data["Residence_type"].unique()
        )

        input_data["avg_glucose_level"] = st.slider(
            "Average Glucose Level",
            min_value=data["avg_glucose_level"].min(),
            max_value=data["avg_glucose_level"].max(),
            value=100.0,
        )

        input_data["bmi"] = str(
            st.slider(
                "BMI",
                min_value=float(data["bmi"].min()),
                max_value=float(data["bmi"].max()),
                value=25.0,
            )
        )

        input_data["smoking_status"] = st.selectbox(
            "Smoking Status", data["smoking_status"].unique()
        )
        input_data = pl.DataFrame(input_data, schema=X_train.schema)

        # due to some errors in custom transformers (not fitting the data), additional rows need to be added
        X_for_prediction = X_train[0:10].vstack(input_data)

        st.dataframe(X_for_prediction[-1])
        y_pred = model_obj.predict_proba(X_for_prediction)[-1]
        st.header(f"Stroke prediction: {y_pred[1]:.2f}")
        st.write(f"No stroke: {y_pred[0]:.2f}")
        st.write(
            "note: only age, avg_glucose_level, bmi, hypertension, heart_disease are used in the prediction. Other features were determined to be less important. "
            "Full selection is here only for consistency."
        )
