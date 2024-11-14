# custom modules/ constants
from custom_modules.constants import DATASET_NAME, DATASET_PATH
from custom_modules.constants import BASE_THEME, ALT_THEME, DISCRETE_COLOR_MAP
from custom_modules.constants import RNG

from custom_modules.custom_transformers import (
    CustomNormalizer,
    NumericImputer,
    NumericBinner,
)
from custom_modules.custom_transformers import (
    NumericScaler,
    BinaryConverter,
    FeaturePreprocessor,
)
from custom_modules.custom_transformers import (
    OneHotEncoderWrapper,
    BasicFeatureCombiner,
)
from custom_modules.custom_transformers import ModelWrapper


# base imports
import kagglehub
import os
import polars as pl
import numpy as np
import time

# visuals
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# environment setup
pio.templates.default = BASE_THEME
px.defaults.template = BASE_THEME
pio.templates[BASE_THEME].layout.colorway = DISCRETE_COLOR_MAP
pio.renderers.default = "notebook_connected"


# scipy
import scipy.stats as stats

# sklearn
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    Normalizer,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector


from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
)

# imbalance-learn
from imblearn.ensemble import BalancedBaggingClassifier


## custom functions
def load_dataset() -> pl.DataFrame:
    """
    Loads the stroke prediction dataset from a local file if it exists,
    otherwise downloads it from Kaggle and saves it locally.
    Returns:
        pl.DataFrame: The loaded dataset as a Polars DataFrame.
    """
    if os.path.isfile(DATASET_PATH):
        print("Dataset already downloaded")
        return pl.read_csv(DATASET_PATH, infer_schema_length=10000)
    else:
        print("Downloading dataset")
        downloaded_path = kagglehub.dataset_download(
            "fedesoriano/stroke-prediction-dataset",
            path=DATASET_NAME,
            force_download=True,
        )
        dataset = pl.read_csv(downloaded_path, infer_schema_length=10000)
        dataset.write_csv(DATASET_PATH)
        return dataset


def get_unique_values(
    data: pl.DataFrame, max_display_count: int = 5, expected_nan: str = "N/A"
) -> pl.DataFrame:
    """
    Generate a summary DataFrame with unique values information for each column in the input DataFrame.
    Parameters:
    -----------
    data : pl.DataFrame
        The input DataFrame to analyze.
    max_display_count : int, optional
        The maximum number of unique values to display for each column (default is 5).
    expected_nan : str, optional
        The string representation of NaN values to check for in the unique values (default is "N/A").
    Returns:
    --------
    pl.DataFrame
        A DataFrame containing the following columns:
        - "Column": The name of the column.
        - "Column Type": The data type of the column.
        - "Unique Values Count": The count of unique values in the column.
        - "Unique Values": A string representation of the unique values or a message indicating more than the max_display_count.
        - "Contains {expected_nan}": A boolean indicating whether the column contains the expected NaN value.
    """
    columns = data.columns
    new_df = []
    for column in columns:
        unique_values = data[column].unique().to_list()
        contains_nan = True if expected_nan in unique_values else False
        unique_values_count = len(unique_values)

        if unique_values_count > max_display_count:
            unique_values = f"More than {max_display_count} unique values"
        else:
            unique_values = ", ".join([str(x) for x in unique_values])

        column_type = data.schema[column]
        new_df.append(
            [column, column_type, unique_values_count, unique_values, contains_nan]
        )

    return pl.DataFrame(
        new_df,
        schema=[
            "Column",
            "Column Type",
            "Unique Values Count",
            "Unique Values",
            f"Contains {expected_nan}",
        ],
        orient="row",
    )


def combine_data(
    X: pl.DataFrame, y: pl.Series, binary_columns: list, y_name: str = "stroke"
) -> pl.DataFrame:
    """
    Combines feature DataFrame with target Series and converts specified binary columns to string representation.

    Parameters:
        X (pl.DataFrame): The feature DataFrame.
        y (pl.Series): The target Series.
        binary_columns (list): List of column names in X that are binary and need to be converted to "Yes"/"No".
        y_name (str, optional): The name for the target column in the combined DataFrame. Defaults to "stroke".

    Returns:
        pl.DataFrame: The combined DataFrame with the target column and binary columns converted to string representation.
    """
    data = X.with_columns((y).alias(y_name))
    for col in binary_columns:
        data = data.with_columns(
            pl.col(col)
            .map_elements(lambda x: "Yes" if x == 1 else "No", return_dtype=pl.String)
            .cast(pl.String)
            .alias(col)
        )
    return data


def plot_numerical_outliers(
    data: pl.DataFrame,
    columns: list[str],
    target_column: str = "stroke",
    title: str = "Numerical feature outliers",
    shape: tuple[int, int] = (3, 1),
) -> go.Figure:
    """
    Plots numerical outliers and distributions for specified columns in a given DataFrame.
    Parameters:
    -----------
        data : pl.DataFrame
            The DataFrame containing the data to be plotted.
        columns : list[str]
            A list of column names for which the outliers and distributions will be plotted.
        target_column : str, optional
            The name of the target column used for coloring the plots. Default is "stroke".
        title : str, optional
            The title of the entire plot. Default is "Numerical feature outliers and distribution".
        shape : tuple[int, int], optional
            The shape of the subplot grid as (rows, columns). Default is (3, 1).
    Returns:
    --------
        go.Figure
            A Plotly Figure object containing the subplots of box plots and violin plots for the specified columns.
    """

    fig = make_subplots(
        rows=shape[0],
        cols=shape[1],
        subplot_titles=[f"{col} column outliers" for col in columns],
    )

    if target_column is not None:
        title = f"{title} grouped by {target_column} column"

    legend_set = set()

    for i, col in enumerate(columns):
        inner_fig = px.box(data, x=target_column, y=col, color=target_column)
        for trace in inner_fig["data"]:
            (
                trace.update(showlegend=False)
                if trace.name in legend_set
                else legend_set.add(trace.name)
            )
            fig.add_trace(trace, row=i + 1, col=1)

        inner_fig = px.violin(data, x=target_column, y=col, color=target_column)
        for trace in inner_fig["data"]:
            (
                trace.update(showlegend=False)
                if trace.name in legend_set
                else legend_set.add(trace.name)
            )
            trace.update(opacity=0.5)
            fig.add_trace(trace, row=i + 1, col=1)

        fig.update_xaxes(title_text=target_column, row=i + 1, col=1)
        fig.update_yaxes(title_text=col, row=i + 1, col=1)

    fig.update_layout(
        title=title, height=shape[0] * 400, legend_title_text=target_column
    )
    return fig


def plot_bar_data(
    data: pl.DataFrame,
    cat_columns: list[str],
    target: str,
    shape: tuple[int, int] = (7, 1),
) -> go.Figure:
    """
    Plots bar charts for categorical columns in a DataFrame, grouped by a target column.
    Parameters:
    -----------
        data : pl.DataFrame
            The DataFrame containing the data to be plotted.
        cat_columns : list[str]
            A list of categorical column names to be plotted.
        target : str
            The target column name by which the data will be grouped.
        shape : tuple[int, int], optional
            The shape of the subplot grid (rows, columns). Default is (7, 1).
    Returns:
    --------
        go.Figure
            A Plotly Figure object containing the bar charts.
    """
    fig = make_subplots(
        rows=shape[0],
        cols=shape[1],
        subplot_titles=[f"{col} column counts" for col in cat_columns],
    )

    colors = dict()
    for i, option in enumerate(data[target].unique().to_list()):
        colors[option] = DISCRETE_COLOR_MAP[i]

    legend_set = set()

    for option in data[target].unique().to_list():
        for i, column in enumerate(cat_columns):
            grouped = data.filter(pl.col(target) == option).group_by(column).len()
            inner_fig = px.bar(grouped, x=column, y="len", color=column, text_auto=True)

            for trace in inner_fig.data:
                trace.update(marker_color=colors[option])
                trace.update(name=option, legendgroup=option)
                (
                    trace.update(showlegend=False)
                    if option in legend_set
                    else legend_set.add(option)
                )
                fig.add_trace(trace, row=i + 1, col=1)
            fig.update_xaxes(title_text=column, row=i + 1, col=1)
            fig.update_yaxes(title_text="Count", row=i + 1, col=1)

    title = f"Categorical features grouped by `{target}` column"
    fig.update_layout(
        title=title, height=shape[0] * 400, legend_title_text=target, barmode="stack"
    )
    return fig


def point_confidence_interval(
    target_count: int, other_count: int, confidence_level: float = 0.95
) -> tuple[float, tuple[float, float]]:
    """
    Calculate the point estimate and confidence interval for a proportion.

    Parameters:
        target_count (int): The count of the target occurrences.
        other_count (int): The count of the other occurrences.
        confidence_level (float, optional): The confidence level for the interval. Default is 0.95.

    Returns:
        tuple: A tuple containing the point estimate (float) and the confidence interval (tuple of two floats).
    """
    total_count = target_count + other_count
    if total_count == 0:
        return (0.0, (0.0, 0.0))

    p_hat = target_count / total_count
    standard_error = np.sqrt((p_hat * (1 - p_hat)) / total_count)
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    margin_of_error = z_score * standard_error
    confidence_interval = (p_hat - margin_of_error, p_hat + margin_of_error)
    return (p_hat, confidence_interval)


def plot_points_subplots(
    data: pl.DataFrame,
    target: str,
    answer: str,
    confidence_level: float,
    features: list[str],
) -> go.Figure:
    """
    Plots point estimates and confidence intervals for specified features in the given dataset.
    Parameters:
    -----------
        data : pl.DataFrame
            The input data containing the features and target variable.
        target : str
            The target variable for which the point estimates and confidence intervals are calculated.
        answer : str
            The specific value of the target variable to be analyzed.
        confidence_level : float
            The confidence level for the confidence intervals (e.g., 0.95 for 95% confidence).
        features : list[str]
            A list of feature names to be analyzed and plotted.
    Returns:
    --------
        go.Figure
            A Plotly Figure object containing the subplots of point estimates and confidence intervals for each feature.
    """
    fig = make_subplots(rows=len(features), cols=1)

    for i, feature in enumerate(features):
        for ans in data[feature].unique().to_list():
            filtered_df = data.filter(pl.col(feature) == ans)
            yes_count = filtered_df.filter(pl.col(target) == answer)[target].len()
            else_count = filtered_df.filter(pl.col(target) != answer)[target].len()

            confidence_result = point_confidence_interval(
                yes_count, else_count, confidence_level
            )
            p_hat, confidence_interval = confidence_result

            trace = go.Scatter(
                x=[ans, ans],
                y=[confidence_interval[0], confidence_interval[1]],
                mode="lines",
                line=dict(color=DISCRETE_COLOR_MAP[1], width=2),
                opacity=0.3,
                name=f"{ans} Confidence Interval",
                showlegend=False,
            )

            fig.add_trace(trace, row=i + 1, col=1)

            trace = go.Scatter(
                x=[ans],
                y=[p_hat],
                mode="lines+markers",
                line=dict(color=DISCRETE_COLOR_MAP[1], width=2),
                name=f"{ans} point percentage",
                showlegend=False,
            )
            fig.add_trace(trace, row=i + 1, col=1)
        fig.update_xaxes(title_text=feature, row=i + 1, col=1)

    fig.update_yaxes(title_text=f"{target} ({answer}) percentage", row=i + 1, col=1)

    fig.update_layout(
        title_text=f"Point Estimate and Confidence Interval ({confidence_level * 100}%)",
        height=400 * len(features),
        hoverlabel_namelength=-1,
        xaxis_categoryorder="total descending",
    )

    fig.update_xaxes(categoryorder="total descending")
    fig.update_yaxes(title_text=f"{target} ({answer}) percentage")

    return fig


def plot_correlation_heatmap(
    data: pl.DataFrame,
    title: str = "Correlation Heatmap",
    sub_title: str = "Continous - Continous Features",
    color_scale: str = "RdBu",
    zmin: int = -1,
    zmax: int = 1,
) -> go.Figure:
    """
    Plots a correlation heatmap using the provided data.

    Parameters:
    -----------
        data : pl.DataFrame
            The data frame containing the correlation matrix to be visualized.
        title : str, optional
            The main title of the heatmap. Default is "Correlation Heatmap".
        sub_title : str, optional
            The subtitle of the heatmap. Default is "Continous - Continous Features".
        color_scale : str, optional
            The color scale to be used for the heatmap. Default is 'RdBu'. Alternatives are 'Viridis'
        zmin : int, optional
            The minimum value for the color scale. Default is -1.
        zmax : int, optional
            The maximum value for the color scale. Default is 1.
    Returns:
    --------
        go.Figure
            The Plotly figure object representing the correlation heatmap.
    """

    fig = px.imshow(
        data,
        color_continuous_scale=color_scale,
        text_auto=True,
        y=data.columns,
        x=data.columns,
        zmax=zmax,
        zmin=zmin,
    )

    fig.update_xaxes(side="top")
    fig.update_layout(
        title=dict(text=title, subtitle=dict(text=sub_title)),
        xaxis_title="Features",
        yaxis_title="Features",
    )
    return fig


def continous_continous_correlation_matrix(
    df: pl.DataFrame, method: str = "spearman"
) -> pl.DataFrame:
    """
    Computes the correlation matrix for continuous variables in a DataFrame using the specified method.

    Parameters:
    df : pl.DataFrame
        The input DataFrame containing continuous variables.
    method : str, optional
        The method to compute the correlation coefficient. Default is "spearman".
        Other possible values include "pearson".

    Returns:
    pl.DataFrame
        A DataFrame containing the correlation matrix with the same columns as the input DataFrame.

    """
    data = []
    for col in df.columns:
        col_list = []
        for col2 in df.columns:
            corr_coef = df.select(pl.corr(col, col2, method=method))[col][0]
            col_list.append(corr_coef)
        data.append(col_list)
    return pl.DataFrame(data, schema=df.columns)


def plot_numeric_scatter(
    data: pl.DataFrame, columns: list[str], target_col: str
) -> go.Figure:
    """
    Plots a scatter plot for two numeric columns in the given DataFrame, colored by the target column.
    Args:
        data (pl.DataFrame): The DataFrame containing the data to plot.
        columns (list[str]): A list of two column names to be used for the x and y axes.
        target_col (str): The column name to be used for coloring the scatter plot points.
    Returns:
        go.Figure: A Plotly Figure object representing the scatter plot.
    """
    fig = px.scatter(
        data_frame=data,
        x=columns[0],
        y=columns[1],
        color=target_col,
        title="Numeric Feature Scatter Plot",
    )

    sub_f_string = f"Relationship between {columns[0]} and {columns[1]}<br>"
    sub_f_string += f"Color represents {target_col}"

    fig.update_layout(title_subtitle_font_size=12, title_subtitle_text=sub_f_string)

    return fig


def categorical_categorical_correlation_matrix(data: pl.DataFrame, method="cramer"):
    """
    Computes the correlation matrix for categorical variables in a DataFrame using the specified method.
    Parameters:
    -----------
    data : pl.DataFrame
        The input DataFrame containing categorical variables.
    method : str, optional
        The method to compute the association measure. Default is "cramer".
        Other possible values can be methods supported by `ss.contingency.association`.
    Returns:
    --------
    pl.DataFrame
        A DataFrame representing the correlation matrix of the categorical variables.
        The diagonal elements are 1.0, indicating perfect correlation with themselves.
    """
    columns = data.columns

    corr_matrix = []
    for col_a in columns:
        col_a_list = []
        for col_b in columns:
            if col_a != col_b:
                partial_df = (
                    data[col_a, col_b]
                    .group_by([col_a, col_b])
                    .agg(pl.len())
                    .pivot(col_a, index=col_b)
                )
                partial_columns = partial_df.columns[1:]
                partial_df = partial_df[partial_columns]
                partial_df = partial_df.fill_null(0)
                corr_coef = stats.contingency.association(partial_df, method=method)
                col_a_list.append(corr_coef)
            else:
                col_a_list.append(1.0)
        corr_matrix.append(col_a_list)
    return pl.DataFrame(corr_matrix, schema=columns)


def plot_catplot(
    data: pl.DataFrame,
    cat_columns: list[str],
    target_column: str,
    target_option: str,
    confidence_level,
) -> go.Figure:
    """
    Plots a categorical plot with confidence intervals for the given data.
    Parameters:
    -----------
        data : pl.DataFrame
            The input data frame containing the categorical and target columns.
        cat_columns : list[str]
            A list of two categorical column names to be plotted on the x and y axes.
        target_column : str
            The name of the target column to be analyzed.
        target_option : str
            The specific value of the target column to be considered for the plot.
        confidence_level : float
            The confidence level for the confidence intervals.
    Returns:
    --------
        go.Figure
            A Plotly Figure object containing the categorical plot with confidence intervals.
    """
    fig = go.Figure()

    cat_x_options = data[cat_columns[0]].unique().to_list()
    cat_y_options = data[cat_columns[1]].unique().to_list()
    legend_set = set()

    for i, y_option in enumerate(cat_y_options):
        for x_option in cat_x_options:
            data_subset = data.filter(
                (pl.col(cat_columns[0]) == x_option)
                & (pl.col(cat_columns[1]) == y_option)
            )
            target_count = data_subset.filter(
                pl.col(target_column) == target_option
            ).shape[0]
            other_count = data_subset.filter(
                pl.col(target_column) != target_option
            ).shape[0]

            ci_result = point_confidence_interval(
                target_count, other_count, confidence_level
            )
            p_hat, confidence_interval = ci_result

            trace = go.Scatter(
                x=[x_option, x_option],
                y=[confidence_interval[0], confidence_interval[1]],
                mode="lines",
                line=dict(color=DISCRETE_COLOR_MAP[i], width=2),
                opacity=0.3,
                name=f"{cat_columns[1]}: {y_option}",
                legendgroup=f"{cat_columns[1]}: {y_option}",
                showlegend=True,
            )

            (
                trace.update(showlegend=False)
                if trace.legendgroup in legend_set
                else legend_set.add(trace.legendgroup)
            )
            fig.add_trace(trace)

            trace = go.Scatter(
                x=[x_option],
                y=[p_hat],
                legendgroup=f"{cat_columns[1]}: {y_option}",
                showlegend=True,
                name=f"{cat_columns[1]}: {y_option}",
                marker=dict(color=DISCRETE_COLOR_MAP[i]),
            )

            (
                trace.update(showlegend=False)
                if trace.legendgroup in legend_set
                else legend_set.add(trace.legendgroup)
            )
            fig.add_trace(trace)

    fig.update_layout(
        title=f"{cat_columns[0]} vs {cat_columns[1]}",
        title_subtitle_text=f"Confidence Level: {confidence_level}",
        hoverlabel_namelength=-1,
        legend_title=f"{cat_columns[1]}",
        xaxis_title=cat_columns[0],
        yaxis_title=f"{target_column} ({target_option}) percentage",
        xaxis_categoryorder="total descending",
    )

    return fig


def upsample_minority_class(
    data: pl.DataFrame, target_column: str, target_class: str
) -> pl.DataFrame:
    """
    Upsamples the minority class in the given DataFrame to match the number of samples in the majority class.
    Parameters:
        data (pl.DataFrame): The input DataFrame containing the data to be upsampled.
        target_column (str): The name of the column containing the target class labels.
        target_class (str): The value of the target class that needs to be upsampled.
    Returns:
        pl.DataFrame: A new DataFrame with the minority class upsampled to match the majority class.
    """
    df_majority = data.filter(pl.col(target_column) != target_class)
    df_minority = data.filter(pl.col(target_column) == target_class)

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=df_majority.shape[0], random_state=RNG
    )

    df_upsampled = pl.concat([df_majority, df_minority_upsampled])
    return df_upsampled


def plot_continous_categorical_correlation_matrix(
    data: pl.DataFrame, categorical_columns: list, continous_columns: list, title: str
) -> go.Figure:
    """
    Plots a correlation matrix between continuous and categorical variables using the Point-Biserial correlation method.
    Args:
        data (pl.DataFrame): The input dataframe containing the data.
        categorical_columns (list): A list of column names representing categorical variables.
        continous_columns (list): A list of column names representing continuous variables.
        title (str): The title of the plot.
    Returns:
        go.Figure: A Plotly figure object representing the correlation matrix.
    Notes:
        - Categorical columns with more than 2 unique values are ignored.
        - The correlation matrix is visualized using a heatmap with a color scale ranging from -1 to 1.
        - The x-axis represents continuous features, and the y-axis represents categorical features.
    """
    cat_columns = dict()
    for cat_col in categorical_columns:
        if data[cat_col].unique().count() <= 2:
            label_encoders = LabelEncoder()
            cat_columns[cat_col] = label_encoders.fit_transform(data[cat_col])

    corr_matrix = []
    for cat_col, cat_values in cat_columns.items():
        cat_col_list = []
        for cont_col in continous_columns:
            corr_stat, corr_p = stats.pointbiserialr(cat_values, data[cont_col])
            cat_col_list.append(corr_stat)
        corr_matrix.append(cat_col_list)

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu",
        text_auto=True,
        y=list(cat_columns.keys()),
        x=continous_columns,
        zmax=1,
        zmin=-1,
    )

    fig.update_xaxes(side="top")
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(
                text="Continous - Categorical Correlation, method=PointBiserial"
            ),
        ),
        xaxis_title="Continous Features",
        yaxis_title="Categorical Features",
    )
    return fig


def plot_violin(data: pl.DataFrame, target_column: str, columns: list[str]):
    """
    Generates a violin plot to display the distribution of two specified columns for each class in the target column.
    Args:
        data (pl.DataFrame): The input data frame containing the data to be plotted.
        target_column (str): The name of the column containing the target classes.
        columns (list[str]): A list containing the names of the two columns to be plotted.
    Returns:
        plotly.graph_objs._figure.Figure: A Plotly Figure object containing the violin plot.
    Example:
        fig = plot_violin(data, 'stroke', ['bmi', "ever_married"])
        fig.show()
    """
    title = f"Violin Plot displaying the distribution of {columns[0]} and {columns[1]} for each {target_column} class"
    target_classes = data[target_column].unique().to_list()

    fig = go.Figure()
    for i, target_class in enumerate(target_classes):
        filtered_data = data.filter(pl.col(target_column) == target_class)
        side = "negative" if i == 0 else "positive"
        trace = go.Violin(
            x=filtered_data[columns[0]],
            y=filtered_data[columns[1]],
            points="all",
            jitter=0.5,
            pointpos=-0.5 + i,
            side=side,
            name=f"{target_column} {target_class}",
            marker=dict(color=DISCRETE_COLOR_MAP[i], opacity=0.2),
            showlegend=True,
            legendgroup=f"{target_column} {target_class}",
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        xaxis_title=columns[0],
        yaxis_title=columns[1],
        legend_title=target_column,
        hoverlabel_namelength=-1,
    )

    return fig


def shapiro_wilk_normality_test_matrix(
    data: pl.DataFrame, target_column: str = "stroke"
) -> pl.DataFrame:
    """
    Perform Shapiro-Wilk normality test on each column of the DataFrame, grouped by the target column.
    Parameters:
    -----------
    data : pl.DataFrame
        The input DataFrame containing the data to be tested.
    target_column : str, optional
        The name of the target column used for grouping the data. Default is "stroke".
    Returns:
    --------
    pl.DataFrame
        A DataFrame containing the results of the Shapiro-Wilk normality test for each column and each group in the target column.
        The DataFrame has the following columns:
        - target_column: The unique values from the target column.
        - Column: The name of the column being tested.
        - Statistic: The test statistic from the Shapiro-Wilk test.
        - PValue: The p-value from the Shapiro-Wilk test.
    """
    columns_to_test = data.columns
    columns_to_test.remove(target_column)

    results = []

    for filter_class in data[target_column].unique():
        for column in columns_to_test:
            result = stats.shapiro(
                data.filter(pl.col(target_column) == filter_class)[column].to_numpy()
            )
            results.append([filter_class, column, result.statistic, result.pvalue])

    results_df = pl.DataFrame(
        results, schema=[target_column, "Column", "Statistic", "PValue"], orient="row"
    )
    return results_df


def levene_variance_test(
    data: pl.DataFrame,
    columns_to_test: list[str],
    target_column: str,
    target_classes: list[str],
    center: str = "median",
) -> pl.DataFrame:
    """
    Perform Levene's test for equal variances on specified columns of a DataFrame.
    Parameters:
        data (pl.DataFrame): The input DataFrame containing the data to be tested.
        columns_to_test (list[str]): List of column names to perform the test on.
        target_column (str): The name of the target column used to split the data into groups.
        target_classes (list[str]): List containing the two target classes to compare.
        center (str, optional): The type of center to use for the test. Default is "median".
    Returns:
        pl.DataFrame: A DataFrame containing the results of the Levene's test with columns:
                    - "Column": The name of the column tested.
                    - "Statistic": The test statistic.
                    - "PValue": The p-value of the test.
    """
    results = []

    a_sample = data.filter(pl.col(target_column) == target_classes[0])
    b_sample = data.filter(pl.col(target_column) == target_classes[1])

    for column in columns_to_test:
        if column == target_column:
            continue

        result = stats.levene(
            a_sample[column].to_numpy(), b_sample[column].to_numpy(), center=center
        )
        results.append([column, result.statistic, result.pvalue])
    return pl.DataFrame(results, schema=["Column", "Statistic", "PValue"], orient="row")


def t_test(
    data: pl.DataFrame,
    columns_to_test: list[str],
    target_column: str,
    target_classes: list[str],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> pl.DataFrame:
    """
    Perform a t-test for the means of two independent samples for each specified column.
    Parameters:
        data (pl.DataFrame): The input dataframe containing the data.
        columns_to_test (list[str]): List of column names to perform the t-test on.
        target_column (str): The name of the column containing the target classes.
        target_classes (list[str]): List containing the two target classes to compare.
        equal_var (bool, optional): If True, perform a standard independent 2 sample test that assumes equal population variances. Defaults to False.
        alternative (str, optional): Defines the alternative hypothesis. The following options are available:
            'two-sided': the means of the two samples are unequal (default).
            'greater': the mean of the first sample is greater than the mean of the second sample.
            'less': the mean of the first sample is less than the mean of the second sample.
    Returns:
        pl.DataFrame: A dataframe containing the results of the t-tests with columns "Column", "Statistic", and "PValue".
    """
    results = []

    a_sample = data.filter(pl.col(target_column) == target_classes[0])
    b_sample = data.filter(pl.col(target_column) == target_classes[1])

    for column in columns_to_test:
        if column == target_column:
            continue

        result = stats.ttest_ind(
            a_sample[column].to_numpy(),
            b_sample[column].to_numpy(),
            equal_var=equal_var,
            random_state=RNG,
            alternative=alternative,
        )
        results.append([column, result.statistic, result.pvalue])
    return pl.DataFrame(results, schema=["Column", "Statistic", "PValue"], orient="row")


def means_difference_and_ci(
    data: pl.DataFrame,
    columns_to_test: list[str],
    target_column: str,
    target_classes: list[str],
    confidence_level: float = 0.95,
) -> pl.DataFrame:
    """
    Calculate the difference in means and confidence intervals for specified columns between two target classes.
    Parameters:
        data (pl.DataFrame): The input data as a Polars DataFrame.
        columns_to_test (list[str]): List of column names to test.
        target_column (str): The column name that contains the target classes.
        target_classes (list[str]): List containing the two target classes to compare.
        confidence_level (float, optional): The confidence level for the confidence interval. Default is 0.95.
    Returns:
        pl.DataFrame: A DataFrame containing the column names, means for each class, difference in means, and the lower and upper bounds of the confidence interval.
    """
    results = []

    a_sample = data.filter(pl.col(target_column) == target_classes[0])
    b_sample = data.filter(pl.col(target_column) == target_classes[1])

    for column in columns_to_test:
        if column == target_column:
            continue

        a_mean = a_sample[column].mean()
        b_mean = b_sample[column].mean()

        a_std = a_sample[column].std()
        b_std = b_sample[column].std()

        a_count = a_sample[column].count()
        b_count = b_sample[column].count()

        SE = np.sqrt((a_std**2 / a_count) + (b_std**2 / b_count))

        dof = (((a_std**2 / a_count) + (b_std**2 / b_count)) ** 2) / (
            ((a_std**2 / a_count) ** 2 / (a_count - 1))
            + ((b_std**2 / b_count) ** 2 / (b_count - 1))
        )

        alpha = 1 - confidence_level

        t_crit = stats.t.ppf(1 - alpha / 2, dof)

        diff_means = a_mean - b_mean
        ci_lower = diff_means - t_crit * SE
        ci_upper = diff_means + t_crit * SE

        results.append([column, a_mean, b_mean, diff_means, ci_lower, ci_upper])

    return pl.DataFrame(
        results,
        schema=[
            "Column",
            f"Mean_A ({target_classes[0]})",
            f"Mean_B ({target_classes[1]})",
            "Difference",
            "CI_Lower",
            "CI_Upper",
        ],
        orient="row",
    )


def chi_square_test_matrix(
    data: pl.DataFrame, target_col: str = "stroke"
) -> pl.DataFrame:
    """
    Perform chi-square tests for homogeneity between each feature in the DataFrame and the target column.
    Parameters:
        data (pl.DataFrame): The input DataFrame containing the features and the target column.
        target_col (str): The name of the target column to test against. Default is "stroke".
    Returns:
        pl.DataFrame: A DataFrame containing the chi-square test results with columns:
                    - "Feature": The name of the feature.
                    - "Chi-Square": The chi-square statistic.
                    - "P-Value": The p-value of the test.
    """
    columns = data.columns
    columns.remove(target_col)

    chi_square_test_results = []

    for column in columns:
        crosstab = stats.contingency.crosstab(
            data[column].to_numpy(), data[target_col].to_numpy()
        )
        chi_result = stats.chi2_contingency(crosstab.count)
        chi_square_test_results.append([column, chi_result[0], chi_result[1]])

    return pl.DataFrame(
        chi_square_test_results,
        schema=["Feature", "Chi-Square", "P-Value"],
        orient="row",
    )


def build_prep_pipeline() -> Pipeline:
    """
    Builds a preprocessing pipeline for stroke prediction data.
    The pipeline consists of the following steps:
    1. Preprocessing for individual columns:
        - 'age': Imputation, binning, and scaling.
        - 'bmi': Imputation, binning, and scaling.
        - 'avg_glucose_level': Imputation, binning, and scaling.
        - 'ever_married': Imputation and binary conversion.
        - 'hypertension': Imputation and binary conversion.
        - 'heart_disease': Imputation and binary conversion.
        - 'work_type': Imputation, one-hot encoding, and dropping the original column.
        - 'Residence_type': Imputation, one-hot encoding, and dropping the original column.
        - 'smoking_status': Imputation, one-hot encoding, and dropping the original column.
        - 'gender': Imputation, one-hot encoding, and dropping the original column.
    2. Binning cleanup for binned columns:
        - 'age_binned': One-hot encoding and dropping the original column.
        - 'avg_glucose_level_binned': One-hot encoding and dropping the original column.
        - 'bmi_binned': One-hot encoding and dropping the original column.
    3. Extra feature generation:
        - 'risk_score_all': Combination of 'age', 'bmi', 'avg_glucose_level', 'hypertension' and 'heart_disease with scaling.
        - 'risk_score_numeric_no_age': Combination of 'bmi' and 'avg_glucose_level' with scaling.
        - 'glucose_bmi_ratio': Ratio of 'avg_glucose_level' to 'bmi' with scaling.
        - 'risk_score_heart': Combination of 'hypertension' and 'heart_disease' with scaling.
        - 'risk_score_numeric': Combination of 'bmi', 'avg_glucose_level', and 'age' with scaling.
    Returns:
        Pipeline: A scikit-learn Pipeline object that performs the preprocessing steps.
    """
    age_pipeline = Pipeline(
        [
            ("imputer", NumericImputer()),
            (
                "binner",
                NumericBinner(
                    bins=[18, 40, 60],
                    bin_labels=["teen", "adult", "middle_age", "senior"],
                ),
            ),
            ("scaler", NumericScaler()),
        ]
    )

    bmi_pipeline = Pipeline(
        [
            ("imputer", NumericImputer()),
            (
                "binner",
                NumericBinner(
                    bins=[18, 25, 30],
                    bin_labels=["underweight", "normal", "overweight", "obese"],
                ),
            ),
            ("scaler", NumericScaler()),
        ]
    )

    glucose_pipeline = Pipeline(
        [
            ("imputer", NumericImputer()),
            (
                "binner",
                NumericBinner(
                    bins=[70, 100, 125],
                    bin_labels=["low-normal", "normal", "pre-diabetic", "diabetic"],
                ),
            ),
            ("scaler", NumericScaler()),
        ]
    )

    ever_married_pipeline = Pipeline(
        [
            (
                "binary_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("binary_converter", BinaryConverter()),
        ]
    )

    hypertension_pipeline = Pipeline(
        [
            (
                "binary_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("binary_converter", BinaryConverter()),
        ]
    )

    heart_disease_pipeline = Pipeline(
        [
            (
                "binary_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("binary_converter", BinaryConverter()),
        ]
    )

    work_type_pipeline = Pipeline(
        [
            (
                "cat_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("one_hot_encoder", OneHotEncoderWrapper()),
            ("drop", {"drop": "work_type"}),
        ]
    )

    residence_type_pipeline = Pipeline(
        [
            (
                "cat_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("one_hot_encoder", OneHotEncoderWrapper()),
            ("drop", {"drop": "Residence_type"}),
        ]
    )

    smoking_status_pipeline = Pipeline(
        [
            (
                "cat_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("one_hot_encoder", OneHotEncoderWrapper()),
            ("drop", {"drop": "smoking_status"}),
        ]
    )

    gender_pipeline = Pipeline(
        [
            (
                "cat_imputer",
                NumericImputer(
                    imputer="simple", strategy="most_frequent", adj_missing_value=False
                ),
            ),
            ("one_hot_encoder", OneHotEncoderWrapper()),
            ("drop", {"drop": "gender"}),
        ]
    )

    age_binned_pipeline = Pipeline(
        [("one_hot_encoder", OneHotEncoderWrapper()), ("drop", {"drop": "age_binned"})]
    )

    glucose_binned_pipeline = Pipeline(
        [
            ("one_hot_encoder", OneHotEncoderWrapper()),
            ("drop", {"drop": "avg_glucose_level_binned"}),
        ]
    )

    bmi_binned_pipeline = Pipeline(
        [("one_hot_encoder", OneHotEncoderWrapper()), ("drop", {"drop": "bmi_binned"})]
    )

    extra_features_pipeline = Pipeline(
        [
            (
                "risk_score_all",
                BasicFeatureCombiner(
                    columns=[
                        "age",
                        "bmi",
                        "avg_glucose_level",
                        "hypertension",
                        "heart_disease",
                    ],
                    combiner_type="add",
                    new_feature_name="risk_score_all",
                    scaler_obj=NumericScaler(),
                ),
            ),
            (
                "risk_score_numeric_no_age",
                BasicFeatureCombiner(
                    columns=["bmi", "avg_glucose_level"],
                    combiner_type="add",
                    new_feature_name="risk_score_no_age",
                    scaler_obj=NumericScaler(),
                ),
            ),
            (
                "glucose_bmi_ratio",
                BasicFeatureCombiner(
                    columns=["avg_glucose_level", "bmi"],
                    combiner_type="divide",
                    new_feature_name="glucose_bmi_ratio",
                    scaler_obj=NumericScaler(),
                ),
            ),
            (
                "risk_score_heart",
                BasicFeatureCombiner(
                    columns=["heart_disease", "hypertension"],
                    combiner_type="add",
                    new_feature_name="risk_score_heart",
                    scaler_obj=NumericScaler(),
                ),
            ),
            (
                "risk_score_numeric",
                BasicFeatureCombiner(
                    columns=["bmi", "avg_glucose_level", "age"],
                    combiner_type="add",
                    new_feature_name="risk_score_numeric",
                    scaler_obj=NumericScaler(),
                ),
            ),
        ]
    )

    prep_pipe = FeaturePreprocessor(
        column_pipelines={
            "age": age_pipeline,
            "bmi": bmi_pipeline,
            "avg_glucose_level": glucose_pipeline,
            "ever_married": ever_married_pipeline,
            "hypertension": hypertension_pipeline,
            "heart_disease": heart_disease_pipeline,
            "work_type": work_type_pipeline,
            "Residence_type": residence_type_pipeline,
            "smoking_status": smoking_status_pipeline,
            "gender": gender_pipeline,
        },
        remaining_columns="drop",
    )

    binning_pipe = FeaturePreprocessor(
        column_pipelines={
            "age_binned": age_binned_pipeline,
            "avg_glucose_level_binned": glucose_binned_pipeline,
            "bmi_binned": bmi_binned_pipeline,
        },
        remaining_columns="passthrough",
    )

    return Pipeline(
        [
            ("preprocessor", prep_pipe),
            ("binned_value_cleanup", binning_pipe),
            ("extra_features", extra_features_pipeline),
        ]
    )


def evaluate_model(
    model, X, y, n_repeats=10, n_splits=5, scoring_metric: str = "recall"
):
    """
    Evaluate a machine learning model using repeated stratified k-fold cross-validation.
    Parameters:
        model : estimator object
            The machine learning model to evaluate. This should be an object that implements the scikit-learn estimator interface.
        X : array-like of shape (n_samples, n_features)
            The input data to fit.
        y : array-like of shape (n_samples,)
            The target variable to try to predict in the case of supervised learning.
        n_repeats : int, default=10
            Number of times cross-validator needs to be repeated.
        n_splits : int, default=5
            Number of folds in each RepeatedStratifiedKFold.
        scoring_metric : str, default="recall"
            A string representing the scoring metric to use for evaluation. This should be a valid scoring metric recognized by scikit-learn.
    Returns:
        scores : array of float, shape=(n_repeats * n_splits,)
            Array of scores of the estimator for each run of the cross-validation.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RNG
    )

    scores = cross_val_score(
        model, X, y, scoring=scoring_metric, cv=cv, n_jobs=-1, error_score="raise"
    )
    return scores


def model_score_matrix(models: dict, X, y, scoring_metric: str = "roc_auc") -> tuple:
    """
    Evaluates multiple machine learning models and returns their performance scores along with a visualization.
    Parameters:
        models (dict): A dictionary where keys are model names and values are the model instances.
        X: Features dataset used for model evaluation.
        y: Target dataset used for model evaluation.
        scoring_metric (str, optional): The metric used for scoring the models. Default is "roc_auc".
    Returns:
        tuple: A tuple containing:
            - scores (pl.DataFrame): A DataFrame with columns ["Model", "Mean <scoring_metric>", "Std <scoring_metric>", "Time Taken"].
            - fig (go.Figure): A plotly Figure object containing box plots of the model scores.
    """
    fig = go.Figure()

    scores = []
    for name, model in models.items():
        start = time.time()
        results = evaluate_model(model, X, y, scoring_metric=scoring_metric)
        end = time.time()
        time_taken = end - start

        scores.append([name, np.mean(results), np.std(results), time_taken])

        trace = go.Box(
            y=results,
            name=f"{name}\nTime taken: {round(time_taken, 2)}s",
            legendgroup=name,
        )
        fig.add_trace(trace)

    scores = pl.DataFrame(
        scores,
        schema=[
            f"Model ({scoring_metric})",
            f"mean_test_score",
            f"test_score_sd",
            "Time Taken",
        ],
        orient="row",
    ).sort("mean_test_score", descending=True)
    fig.update_layout(
        title_text="Various Model Scores",
        template=pio.templates.default,
        title_subtitle_text=f"Scoring metric: {scoring_metric}",
        hoverlabel_namelength=-1,
    )
    return scores, fig


def build_model_list(prep_pipeline: Pipeline) -> dict:
    models = {
        # Ensamble classifiers
        "Random Forest": ModelWrapper(
            RandomForestClassifier(random_state=RNG), prep_pipeline
        ),
        "Gradient Boosting": ModelWrapper(
            GradientBoostingClassifier(random_state=RNG), prep_pipeline
        ),
        # "AdaBoost": ModelWrapper(AdaBoostClassifier(random_state=RNG), prep_pipeline),
        "HistGradientBoosting": ModelWrapper(
            HistGradientBoostingClassifier(random_state=RNG), prep_pipeline
        ),
        # Linear models
        "Logistic Regression": ModelWrapper(
            LogisticRegression(random_state=RNG), prep_pipeline
        ),
        "SGD Classifier": ModelWrapper(SGDClassifier(random_state=RNG), prep_pipeline),
        "Ridge Classifier": ModelWrapper(
            RidgeClassifier(random_state=RNG), prep_pipeline
        ),
        "Gaussian Naive Bayes": ModelWrapper(GaussianNB(), prep_pipeline),
        "Passive Aggressive": ModelWrapper(
            PassiveAggressiveClassifier(random_state=RNG), prep_pipeline
        ),
        "KNNeighbors": ModelWrapper(KNeighborsClassifier(), prep_pipeline),
        "SVC": ModelWrapper(SVC(random_state=RNG), prep_pipeline),
        "Linear SVC": ModelWrapper(LinearSVC(random_state=RNG), prep_pipeline),
        # Base decision tree models
        "Decision Tree": ModelWrapper(
            DecisionTreeClassifier(random_state=RNG), prep_pipeline
        ),
        "Extra Tree": ModelWrapper(
            ExtraTreeClassifier(random_state=RNG), prep_pipeline
        ),
    }
    return models


def evaluate_confusion_matrix(
    model_dict: dict,
    X_train: pl.DataFrame,
    y_train: pl.Series,
    n_splits: int = 3,
    n_repeats: int = 2,
    verbose: bool = False,
):
    """
    Evaluates the confusion matrix for multiple models using repeated stratified k-fold cross-validation.
    Parameters:
        model_dict (dict): A dictionary where keys are model names and values are model pipelines.
        X_train (pl.DataFrame): The training data features.
        y_train (pl.Series): The training data labels.
        n_splits (int, optional): Number of splits for cross-validation. Default is 3.
        n_repeats (int, optional): Number of repeats for cross-validation. Default is 2.
        verbose (bool, optional): If True, prints the mean confusion matrix values for each model. Default is False.
    Returns:
        pl.DataFrame: A DataFrame containing the mean confusion matrix values (True Negatives, False Positives, False Negatives, True Positives) for each model.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RNG
    )
    results = {
        "Observation": [
            "Mean True Negatives",
            "Mean False Positives",
            "Mean False Negatives",
            "Mean True Positives",
        ]
    }
    for model_name, model_pipe in model_dict.items():
        splits = cv.split(X_train, y_train)
        tn_count = 0
        fn_count = 0
        tp_count = 0
        fp_count = 0
        model = model_pipe
        for i, (train_split, val_split) in enumerate(splits):
            train_X = X_train[train_split]
            train_y = y_train[train_split]
            val_X = X_train[val_split]
            val_y = y_train[val_split]
            model.fit(train_X, train_y)
            pred_y = model.predict(val_X)
            c_matrix = confusion_matrix(val_y, pred_y)
            tn, fp, fn, tp = c_matrix.ravel()
            tn_count += tn
            fp_count += fp
            fn_count += fn
            tp_count += tp

        mean_tn = round((tn_count / (n_splits * n_repeats)), 2)
        mean_fp = round((fp_count / (n_splits * n_repeats)), 2)
        mean_fn = round((fn_count / (n_splits * n_repeats)), 2)
        mean_tp = round((tp_count / (n_splits * n_repeats)), 2)

        total = mean_tn + mean_fp + mean_fn + mean_tp
        total = round(total, 2)

        results[model_name] = [
            f"{mean_tn} ({round(mean_tn/total, 2)})",
            f"{mean_fp} ({round(mean_fp/total, 2)})",
            f"{mean_fn} ({round(mean_fn/total, 2)})",
            f"{mean_tp} ({round(mean_tp/total, 2)})",
        ]

        if verbose:
            print(f"Model: {model_name}")
            print(f"Mean True Negatives: {mean_tn}")
            print(f"Mean False Positives: {mean_fp}")
            print(f"Mean False Negatives: {mean_fn}")
            print(f"Mean True Positives: {mean_tp}")

    return pl.DataFrame(results)


def select_k_best(
    X: pl.DataFrame, y: pl.Series, k: int, score_func: callable = f_classif
) -> pl.DataFrame:
    """
    Selects the top k features from the input DataFrame X based on their
    ANOVA F-value with the target variable y.
    Parameters:
        X (pl.DataFrame): The input DataFrame containing the features.
        y (pl.Series): The target variable.
        k (int): The number of top features to select.
    Returns:
        pl.DataFrame: A DataFrame containing the selected top k features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = selector.get_feature_names_out()

    return pl.DataFrame(X_selected, schema=list(selected_columns))


def evaluate_k_best(
    model_dict: dict, X: pl.DataFrame, y: pl.Series, scoring: str
) -> pl.DataFrame:
    """
    Evaluates multiple models using the k-best feature selection method and returns the results.
    Parameters:
        model_dict (dict): A dictionary where keys are model names and values are model objects.
        X (pl.DataFrame): The input features as a DataFrame.
        y (pl.Series): The target variable as a Series.
        scoring (str): The scoring metric to evaluate the models.
    Returns:
        pl.DataFrame: A DataFrame containing the base mean score, k-best mean score, and the number of k-best features for each model.
    """
    results = {
        f"Observations {scoring}": [
            "Base Mean Score",
            "K-Best Mean Score",
            "k-best Features",
        ]
    }
    prep = build_prep_pipeline()

    x_prepped = prep.fit_transform(X)
    x_col_len = x_prepped.shape[1]

    for name, model in model_dict.items():
        base_results = float(
            np.mean(evaluate_model(model.model, x_prepped, y, scoring_metric=scoring))
        )
        best_result = None
        best_k = None
        for i in range(1, x_col_len + 1):

            X_k_best_cols = select_k_best(x_prepped, y, i).columns
            k_best_result = float(
                np.mean(
                    evaluate_model(
                        model.model, x_prepped[X_k_best_cols], y, scoring_metric=scoring
                    )
                )
            )

            if best_result is None:
                best_result = k_best_result
                best_k = i
            elif k_best_result > best_result:
                best_result = k_best_result
                best_k = i
        results[name] = [base_results, best_result, best_k]

    return pl.DataFrame(results)


def rebuild_balanced_models(prep_pipeline) -> dict:
    """
    Rebuilds a dictionary of models by wrapping each model in a BalancedBaggingClassifier
    to handle imbalanced datasets.
    Args:
        model_dict (dict): A dictionary where keys are model names and values are model instances.
    Returns:
        dict: A dictionary with the same keys as the input, but with each model wrapped in a
              BalancedBaggingClassifier to ensure balanced training.
    """
    model_list = build_model_list(prep_pipeline)
    for name, model in model_list.items():
        new_model = ModelWrapper(
            BalancedBaggingClassifier(
                estimator=model.model,
                sampling_strategy="auto",
                replacement=False,
                random_state=RNG,
            ),
            prep_pipeline=prep_pipeline,
        )

        model_list[name] = new_model
    return model_list


def evaluate_sfs(
    model_dict: dict,
    prep_pipeline: Pipeline,
    X: pl.DataFrame,
    y: pl.Series,
    direction: str = "forward",
    scoring: str = "recall",
) -> tuple:
    """
    Evaluates models using Sequential Feature Selector (SFS) and returns the selected features for each model.
    Parameters:
        model_dict (dict): A dictionary where keys are model names and values are model instances.
        prep_pipeline (Pipeline): A preprocessing pipeline to transform the input features.
        X (pl.DataFrame): The input features as a pandas DataFrame.
        y (pl.Series): The target variable as a pandas Series.
        direction (str, optional): The direction of feature selection, either "forward" or "backward". Default is "forward".
        scoring (str, optional): The scoring metric to evaluate feature subsets. Default is "recall".
    Returns:
        tuple: A tuple containing two DataFrames. The first DataFrame contains the selected features for each model and the second DataFrame contains the model results.
    """
    model_results = {"Metric": ["Feature Count"]}
    features_counter = {}
    prepped_x = prep_pipeline.fit_transform(X)

    for name, model in model_dict.items():
        sfs = None
        sfs = SequentialFeatureSelector(
            estimator=model.model, direction=direction, scoring=scoring, cv=5, n_jobs=-1
        )
        sfs.fit(prepped_x, y)
        features = list(sfs.get_feature_names_out())
        for feature in features:
            if feature in features_counter:
                features_counter[feature] += 1
            else:
                features_counter[feature] = 1
        model_results[name] = [len(features)]

    feature_results = {
        "Feature": list(features_counter.keys()),
        "Count": list(features_counter.values()),
    }
    feature_results = pl.DataFrame(feature_results)
    model_results = pl.DataFrame(model_results)
    return (feature_results, model_results)


def grid_search_tune(model, param_grid, X_train, y_train, scoring="recall") -> tuple:
    """
    Perform a grid search to tune hyperparameters for a given model.

    Parameters:
        model (estimator object): The machine learning model to be tuned.
        param_grid (dict or list of dictionaries): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries.
        X_train (array-like or sparse matrix): The training input samples.
        y_train (array-like): The target values (class labels) as integers or strings.
        scoring (str, default="recall"): A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).

    Returns:
        tuple: A tuple containing the best parameters (dict), the best score (float), and the best estimator (estimator object).
    """
    cv = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=5)
    cv.fit(X_train, y_train)
    return (cv.best_params_, cv.best_score_, cv.best_estimator_)


def build_final_models_dict() -> dict:
    """
    Builds a dictionary of machine learning models wrapped with preprocessing pipelines and their corresponding
    hyperparameter grids for grid search.
    Returns:
        dict: A dictionary where keys are model names and values are dictionaries containing the model wrapped with
              preprocessing pipeline and the hyperparameter grid for grid search.
              Example:
              {
                  "SVC": {
                      "model": ModelWrapper(...),
                      "params": {
                  },
                  ...
    """
    model_dict = {}
    # SVC
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16, 32],
        "model__estimator__C": [0.001, 0.1, 1.0, 10.0],
        "model__estimator__kernel": ["rbf", "poly"],
        "model__estimator__gamma": ["scale", "auto"],
        "model__estimator__degree": [1, 2, 3, 4, 5],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=SVC(random_state=RNG, class_weight="balanced"),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )
    model_dict["SVC"] = {"model": model, "params": param_grid}

    # RidgeClassifier
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16, 32],
        "model__estimator__alpha": [0.001, 0.1, 1.0, 10.0],
        "model__estimator__solver": [
            "auto",
            "svd",
            "cholesky",
            "lsqr",
            "sparse_cg",
            "sag",
            "saga",
        ],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=RidgeClassifier(random_state=RNG, class_weight="balanced"),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )
    model_dict["RidgeClassifier"] = {"model": model, "params": param_grid}

    # Histogram Gradient Boosting
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16],
        "model__estimator__learning_rate": [0.01, 0.1],
        "model__estimator__max_iter": [50, 100, 200],
        "model__estimator__min_samples_leaf": [20, 5, 30],
        "model__estimator__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=HistGradientBoostingClassifier(
                random_state=RNG, class_weight="balanced"
            ),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )

    model_dict["HistGradientBoostingClassifier"] = {
        "model": model,
        "params": param_grid,
    }

    # Perceptron
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16],
        "model__estimator__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
        "model__estimator__penalty": ["l2", "l1", "elasticnet"],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=Perceptron(random_state=RNG, class_weight="balanced", n_jobs=-1),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )

    model_dict["Perceptron"] = {"model": model, "params": param_grid}

    # PassiveAggressiveClassifier
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16],
        "model__estimator__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__estimator__loss": ["hinge", "squared_hinge"],
        "model__estimator__average": [True, False],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=PassiveAggressiveClassifier(
                random_state=RNG, class_weight="balanced", n_jobs=-1
            ),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )

    model_dict["PassiveAggressiveClassifier"] = {"model": model, "params": param_grid}

    # DecisionTreeClassifier
    final_prep_pipeline = build_prep_pipeline()
    final_prep_pipeline.steps.append(("k_best", SelectKBest(f_classif, k=2)))

    param_grid = {
        "prep_pipeline__k_best__k": [1, 2, 4, 8, 16],
        "model__estimator__criterion": ["gini", "entropy", "log_loss"],
        "model__estimator__splitter": ["best", "random"],
        "model__estimator__min_samples_split": [2, 4],
        "model__estimator__min_samples_leaf": [1, 2, 4],
        "model__estimator__max_features": [None, "sqrt", "log2"],
    }

    model = ModelWrapper(
        BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RNG, class_weight="balanced"),
            sampling_strategy="auto",
            replacement=True,
            random_state=RNG,
        ),
        prep_pipeline=final_prep_pipeline,
    )

    model_dict["DecisionTreeClassifier"] = {"model": model, "params": param_grid}

    return model_dict


def hypertune_models(
    model_dict: dict,
    X,
    y,
    scoring: str = "recall",
    param_grid_overwrite: dict = None,
    verbose: bool = False,
    search_type: str = "grid",
) -> dict:
    """
    Hypertune multiple models using grid search.
    Parameters:
        model_dict (dict): A dictionary where keys are model names and values are dictionaries with 'model' and 'params' keys.
                        'model' is the machine learning model instance and 'params' is the parameter grid for tuning.
        X: Features dataset used for training the models.
        y: Target dataset used for training the models.
        scoring (str, optional): Scoring metric to evaluate the models. Default is "recall".
        param_grid_overwrite (dict, optional): Custom parameter grid to overwrite the default parameter grid in model_dict. Default is None.
        verbose (bool, optional): If True, prints detailed logs during the tuning process. Default is False.
        search_type (str, optional): Type of search to perform. Currently, only "grid" search is supported. Default is "grid".
    Returns:
        dict: A dictionary where keys are model names and values are dictionaries with 'model', 'score', and 'params' keys.
            'model' is the best tuned model instance, 'score' is the best score achieved, and 'params' are the best parameters found.
    """
    tuned_models = {}
    if verbose:
        print("Starting hypertuning")
        print(f"Scoring metric: {scoring}")
        print(f"Search type: {search_type}")
        if param_grid_overwrite is not None:
            print("Using custom param grid")

    for model_name, model_data in model_dict.items():
        model = model_data["model"]
        if param_grid_overwrite is not None:
            param_grid = param_grid_overwrite[model_name]
        else:
            param_grid = model_data["params"]

        if verbose:
            print(f"Starting tuning for {model_name}")
        start = time.time()
        match search_type:
            case "grid":
                best_params, best_score, best_model = grid_search_tune(
                    model, param_grid, X, y, scoring
                )
            case _:
                raise ValueError(f"Invalid search type: {search_type}")

        end = time.time()
        if verbose:
            print(f"Finished tuning for {model_name} in {end - start} seconds")
            print(f"Best score: {best_score}")

        tuned_models[model_name] = {
            "model": best_model,
            "score": best_score,
            "params": best_params,
        }

    return tuned_models


def plot_precision_recall_curve(model, X, y):
    """
    Plots the Precision-Recall curve for a given model and dataset.
    Parameters:
        model (sklearn.base.BaseEstimator): The trained model with a predict_proba method.
        X (array-like or sparse matrix): The input samples.
        y (array-like): The true labels.
    Returns:
        plotly.graph_objs._figure.Figure: The plotly figure object containing the Precision-Recall curve.
    """
    y_score = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_score)

    fig = px.line(
        x=recall,
        y=precision,
        title="Precision-Recall Curve",
        labels=dict(x="Recall", y="Precision"),
    )
    precision = precision + 0.0001  # to avoid division by zero
    recall = recall + 0.0001  # to avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]

    trace = go.Scatter(
        x=[recall[best_threshold_index]],
        y=[precision[best_threshold_index]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name=f"Best F1 Score ({best_f1:.2f})",
    )
    fig.add_trace(trace)
    return fig
