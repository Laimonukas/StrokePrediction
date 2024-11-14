from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    Normalizer,
    OneHotEncoder,
    FunctionTransformer,
    LabelEncoder,
    StandardScaler,
)


from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
)

import numpy as np
import polars as pl


class CustomNormalizer(BaseEstimator, TransformerMixin):
    """
    CustomNormalizer is a scikit-learn transformer that normalizes the input data to a range between 0 and 1.
    Methods
    -------
    transform(X)
        Normalizes the input data X to a range between 0 and 1.
    fit(X, y=None)
        Fits the transformer to the data. This transformer does not learn from the data, so this method just returns self.
    """

    def __init__(self):
        pass

    def transform(self, X):
        max = self.max
        min = self.min
        if min == max:
            X = X
        else:
            X = (X - min) / (max - min)
        return X

    def fit(self, X, y=None):
        self.max = np.max(X)
        self.min = np.min(X)
        return self


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for imputing missing numeric values in a dataset.
    Parameters
    ----------
    missing_value : str, default="N/A"
        The placeholder for the missing values in the dataset.
    new_value : float, default=-1.0
        The value to replace the missing values with before imputation.
    return_dtype : type, default=np.float64
        The data type to which the transformed data should be cast.
    imputer : str, default="knn"
        The imputation method to use. Options are "knn" for KNNImputer and "simple" for SimpleImputer.
    strategy : str, default="median"
        The imputation strategy to use with SimpleImputer. Options are "mean", "median", "most_frequent", or "constant".
    adj_missing_value : bool, default=True
        Whether to adjust the missing values before imputation.
    Attributes
    ----------
    imputer_obj : object
        The imputer object used for imputing missing values.
    scaler : object
        The scaler object used for scaling data (if applicable).
    Methods
    -------
    remove_missing_values(X)
        Replaces missing values in the dataset with the specified new_value.
    transform(X)
        Transforms the dataset by imputing missing values.
    fit(X, y=None)
        Fits the transformer to the dataset. This transformer does not require fitting.
    """

    def __init__(
        self,
        missing_value: str = "N/A",
        new_value: float = -1.0,
        return_dtype: type = np.float64,
        imputer: str = "knn",
        strategy: str = "median",
        adj_missing_value: bool = True,
    ):
        self.missing_value = missing_value
        self.new_value = new_value
        self.return_dtype = return_dtype
        self.imputer = imputer
        self.strategy = strategy
        self.adj_missing_value = adj_missing_value

        match self.imputer:
            case "knn":
                self.imputer_obj = KNNImputer(missing_values=self.new_value)
            case "simple":
                self.imputer_obj = SimpleImputer(
                    missing_values=self.new_value, strategy=self.strategy
                )
            case _:
                raise ValueError("Invalid imputer value")

    @property
    def missing_value(self):
        return self._missing_value

    @missing_value.setter
    def missing_value(self, value):
        self._missing_value = value

    @property
    def new_value(self):
        return self._new_value

    @new_value.setter
    def new_value(self, value):
        self._new_value = value

    @property
    def return_dtype(self):
        return self._return_dtype

    @return_dtype.setter
    def return_dtype(self, value):
        self._return_dtype = value

    @property
    def imputer(self):
        return self._imputer

    @imputer.setter
    def imputer(self, value):
        self._imputer = value

    @property
    def imputer_obj(self):
        return self._imputer_obj

    @imputer_obj.setter
    def imputer_obj(self, value):
        self._imputer_obj = value

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    @property
    def adj_missing_value(self):
        return self._adj_missing_value

    @adj_missing_value.setter
    def adj_missing_value(self, value):
        self._adj_missing_value = value

    def remove_missing_values(self, X):
        if type(X) == np.ndarray:
            if self.missing_value in np.unique(X):
                if X.dtype == "object":
                    X[X == self.missing_value] = str(self.new_value)
                    X = X.astype(self.return_dtype)
                else:
                    X[X == self.missing_value] = self.new_value
                return X
            else:
                return X.astype(self.return_dtype)

    def transform(self, X):
        if type(X) == pl.DataFrame:
            schema = X.columns
            new_x = None
            for column in schema:
                np_X = X[column].to_numpy()
                np_X = np_X.reshape(-1, 1)
                if self.adj_missing_value:
                    np_X = self.remove_missing_values(np_X)

                np_X = self.imputer_obj.fit_transform(np_X)
                if new_x is None:
                    new_x = np_X
                else:
                    new_x = np.hstack((new_x, np_X))

            return pl.DataFrame(new_x, schema=schema)
        else:
            schema = [X.name]
            np_X = X.to_numpy()
            np_X = np_X.reshape(-1, 1)
            if self.adj_missing_value:
                np_X = self.remove_missing_values(np_X)
            np_X = self.imputer_obj.fit_transform(np_X)
            return pl.DataFrame(np_X, schema=schema)

    def fit(self, X, y=None):
        return self


class NumericScaler(BaseEstimator, TransformerMixin):
    """
    A custom transformer for scaling numeric data using different scaling techniques.
    Parameters
    ----------
    scaler : str, default="normalize"
        The type of scaler to use. Options are:
        - "standard": StandardScaler from sklearn
        - "normalize": CustomNormalizer (a custom normalization class)
    Attributes
    ----------
    scaler : str
        The type of scaler being used.
    scaler_obj : object
        The scaler object instance based on the specified scaler type.
    Methods
    -------
    transform(X)
        Transforms the input DataFrame or Series X using the specified scaler.
    fit(X, y=None)
        Fits the transformer to the data (no-op for this transformer).
    """

    def __init__(self, scaler: str = "normalize"):
        self.scaler = scaler

        match self.scaler:
            case "standard":
                self.scaler_obj = StandardScaler()
            case "normalize":
                self.scaler_obj = CustomNormalizer()
            case _:
                raise ValueError("Invalid scaler value")

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def scaler_obj(self):
        return self._scaler_obj

    @scaler_obj.setter
    def scaler_obj(self, value):
        self._scaler_obj = value

    def transform(self, X):
        if type(X) == pl.DataFrame:
            schema = X.columns
            new_x = None
            for column in schema:
                np_X = X[column].to_numpy()
                np_X = np_X.reshape(-1, 1)

                np_X = self.scaler_obj.fit_transform(np_X)
                if new_x is None:
                    new_x = np_X
                else:
                    new_x = np.hstack((new_x, np_X))

            return pl.DataFrame(new_x, schema=schema)
        else:
            schema = [X.name]
            np_X = X.to_numpy()
            np_X = np_X.reshape(-1, 1)

            np_X = self.scaler_obj.fit_transform(np_X)
            return pl.DataFrame(np_X, schema=schema)

    def fit(self, X, y=None):
        return self


class NumericBinner(BaseEstimator, TransformerMixin):
    """
    A custom transformer for binning numeric data into specified bins with labels.
    Parameters
    ----------
    bins : list of int
        The bin edges for binning the data.
    bin_labels : list of str
        The labels for the bins.
    Attributes
    ----------
    bins : list of int
        The bin edges for binning the data.
    bin_labels : list of str
        The labels for the bins.
    Methods
    -------
    bin_data(X)
        Bins the data in a pandas Series according to the specified bins and labels.
    transform(X)
        Transforms the input DataFrame or Series by binning its numeric columns.
    fit(X, y=None)
        Fits the transformer to the data (no-op for this transformer).
    Raises
    ------
    ValueError
        If bins or bin_labels are None or if they are not lists of integers and strings respectively.
    """

    def __init__(self, bins: list[int] = None, bin_labels: list[str] = None):
        self.bins = bins
        self.bin_labels = bin_labels

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value):
        if value is None:
            raise ValueError("Bins cannot be None")
        elif all(isinstance(x, int) for x in value):
            self._bins = value
        else:
            raise ValueError("Bins must be a list of integers")

    @property
    def bin_labels(self):
        return self._bin_labels

    @bin_labels.setter
    def bin_labels(self, value):
        if value is None:
            raise ValueError("Bin labels cannot be None")
        elif all(isinstance(x, str) for x in value):
            self._bin_labels = value
        else:
            raise ValueError("Bin labels must be a list of strings")

    def bin_data(self, X):
        if type(X) == pl.Series:
            return X.cut(self.bins, labels=self.bin_labels)

    def transform(self, X):
        if type(X) == pl.DataFrame:
            schema = X.columns
            for column in schema:
                X = X.with_columns((self.bin_data(X[column])).alias(f"{column}_binned"))
        if type(X) == pl.Series:
            name = X.name
            return pl.DataFrame([X, self.bin_data(X)], schema=[name, f"{name}_binned"])

        return X

    def fit(self, X, y=None):
        return self


class BinaryConverter(BaseEstimator, TransformerMixin):
    """
    A custom transformer for converting binary categorical features to numerical values.
    This transformer converts columns with "Yes" and "No" values to 1 and 0 respectively.
    If a column does not contain "Yes" and "No" values, it casts the column to integer type.
    Methods
    -------
    __init__():
        Initializes the BinaryConverter instance.
    transform(X):
        Transforms the input DataFrame or Series by converting binary categorical features to numerical values.
        Parameters:
        X : pl.DataFrame or pl.Series
            The input data to transform.
        Returns:
        pl.DataFrame or pl.Series
            The transformed data with binary categorical features converted to numerical values.
    fit(X, y=None):
        Fits the transformer to the data. This transformer does not learn from the data, so this method just returns self.
        Parameters:
        X : pl.DataFrame or pl.Series
            The input data to fit.
        y : None
            Ignored.
        Returns:
        self : BinaryConverter
            The fitted transformer.
    """

    def __init__(self):
        pass

    def transform(self, X):
        if type(X) == pl.DataFrame:
            schema = X.columns
            for column in schema:
                if "Yes" in X[column].unique().to_list():
                    X = X.with_columns(
                        pl.col(column).replace({"Yes": 1, "No": 0}).cast(pl.Int8)
                    )
                else:
                    X = X.with_columns(pl.col(column).cast(pl.Int8))
            return X
        if type(X) == pl.Series:
            if "Yes" in X.unique().to_list():
                return X.replace({"Yes": 1, "No": 0}).cast(pl.Int8)
            else:
                return X.cast(pl.Int8)

        return X

    def fit(self, X, y=None):
        return self


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom feature preprocessor for transforming columns in a DataFrame or Series using specified pipelines.
    Parameters
    ----------
    column_pipelines : dict, optional
        A dictionary where keys are column names and values are lists of transformations (Pipeline steps) to be applied to those columns.
        Example: {'column_name': [Pipeline_step1, Pipeline_step2, ...]}
    remaining_columns : str, optional, default="drop"
        Specifies what to do with columns that are not in the column_pipelines dictionary.
        Options are:
        - "drop": Drop the columns not specified in column_pipelines.
        - "passthrough": Keep the columns not specified in column_pipelines as they are.
    Attributes
    ----------
    column_pipelines : dict
        The dictionary of column transformation pipelines.
    remaining_columns : str
        The strategy for handling columns not specified in column_pipelines.
    Methods
    -------
    fit(X, y=None)
        Fits the preprocessor to the data. This method does nothing and is present for compatibility with scikit-learn's TransformerMixin.
    transform(X)
        Transforms the input DataFrame or Series according to the specified column pipelines and remaining columns strategy.
    Raises
    ------
    ValueError
        If column_pipelines is None or not a dictionary.
        If remaining_columns is not "drop" or "passthrough".
        If an invalid value is encountered for remaining columns during transformation.
    """

    def __init__(self, column_pipelines: dict = None, remaining_columns: str = "drop"):
        self.column_pipelines = column_pipelines
        self.remaining_columns = remaining_columns

    @property
    def column_pipelines(self):
        return self._column_pipelines

    @column_pipelines.setter
    def column_pipelines(self, value):
        if value is None:
            raise ValueError("Column pipelines cannot be None")
        elif type(value) != dict:
            raise ValueError(
                "Column pipelines must be a dictionary(eg. {'column_name': Pipeline})"
            )
        self._column_pipelines = value

    @property
    def remaining_columns(self):
        return self._remaining_columns

    @remaining_columns.setter
    def remaining_columns(self, value):
        if value not in ["drop", "passthrough"]:
            raise ValueError("Invalid value for remaining columns")
        self._remaining_columns = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transform_columns = self.column_pipelines.keys()
        if type(X) == pl.DataFrame:
            columns = X.columns
            for column in columns:
                if column not in transform_columns:
                    if self.remaining_columns == "drop":
                        X = X.drop(column)
                        continue
                    elif self.remaining_columns == "passthrough":
                        continue
                    else:
                        raise ValueError("Invalid value for remaining columns")
                steps = self.column_pipelines[column]
                for step in steps:
                    if type(step) == dict:
                        if "drop" in step.keys():
                            X = X.drop(step["drop"])
                    else:
                        X = X.with_columns(step.transform(X[column]))

        elif type(X) == pl.Series:
            col_name = X.name
            if col_name not in transform_columns:
                if self.remaining_columns == "drop":
                    return pl.DataFrame()
                elif self.remaining_columns == "passthrough":
                    pass
                else:
                    raise ValueError("Invalid value for remaining columns")

            steps = self.column_pipelines[col_name]
            for step in steps:
                if type(step) == dict:
                    if "drop" in step.keys():
                        X = X.drop(step["drop"])
                else:
                    X = step.transform(X)
            return X
        return X


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for the OneHotEncoder that integrates with scikit-learn's
    BaseEstimator and TransformerMixin for use in pipelines.
    Methods
    -------
    __init__():
        Initializes the OneHotEncoderWrapper with a OneHotEncoder instance.
    transform_column(X: pl.Series):
        Transforms a single Polars Series column into a one-hot encoded DataFrame.
        Parameters:
        X (pl.Series): The input Polars Series to be transformed.
        Returns:
        pl.DataFrame: A DataFrame with one-hot encoded columns.
    transform(X):
        Transforms the input data (either a Polars Series or DataFrame) into
        one-hot encoded format.
        Parameters:
        X (Union[pl.Series, pl.DataFrame]): The input data to be transformed.
        Returns:
        pl.DataFrame: A DataFrame with one-hot encoded columns.
    fit(X, y=None):
        Fits the transformer to the data. This method does nothing and is
        present for compatibility with scikit-learn pipelines.
        Parameters:
        X (Union[pl.Series, pl.DataFrame]): The input data to fit.
        y (optional): Ignored.
        Returns:
        self: Returns the instance itself.
    """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)

    def transform_column(self, X: pl.Series):
        name = X.name
        X = X.to_numpy().reshape(-1, 1)
        matrix = self.encoder.fit_transform(X)
        categories = self.encoder.categories_
        names = [f"{name}_{category}" for category in categories[0]]
        return pl.DataFrame(matrix, schema=names).cast(pl.Int8)

    def transform(self, X):
        if type(X) == np.array:
            X = X.reshape(-1, 1)
        if type(X) == pl.Series:
            return self.transform_column(X)
        elif type(X) == pl.DataFrame:
            schema = X.columns
            for column in schema:
                X = X.with_columns(self.transform_column(X[column]))
            return X

    def fit(self, X, y=None):
        return self


class BasicFeatureCombiner(BaseEstimator, TransformerMixin):
    """
    A custom transformer for combining multiple features into a single feature using specified arithmetic operations.
    Parameters
    ----------
    columns : list of str
        List of column names to be combined. Must contain at least two column names.
    combiner_type : str, default="add"
        Type of combination operation to be performed. Must be one of ["add", "subtract", "multiply", "divide"].
    new_feature_name : str
        Name of the new feature created after combining the specified columns.
    scaler_obj : NumericScaler, optional
        An instance of a scaler object to scale the new feature. Must be an instance of NumericScaler.
    Attributes
    ----------
    columns_ : list of str
        Validated list of column names to be combined.
    combiner_type_ : str
        Validated type of combination operation.
    new_feature_name_ : str
        Validated name of the new feature.
    scaler_obj_ : NumericScaler or None
        Validated scaler object.
    Methods
    -------
    fit(X, y=None)
        Fits the transformer to the data. This transformer does not learn from the data, so this method returns self.
    transform(X)
        Transforms the input DataFrame by combining the specified columns using the specified operation and optionally scaling the new feature.
    Raises
    ------
    ValueError
        If any of the parameters are invalid.
    """

    def __init__(
        self,
        columns: list[str] = None,
        combiner_type: str = "add",
        new_feature_name: str = None,
        scaler_obj: NumericScaler = None,
    ):
        self.columns = columns
        self.combiner_type = combiner_type
        self.new_feature_name = new_feature_name
        self.scaler_obj = scaler_obj

    @property
    def combiner_type(self):
        return self._combiner_type

    @combiner_type.setter
    def combiner_type(self, value):
        if value not in ["add", "subtract", "multiply", "divide"]:
            raise ValueError("Invalid combiner type")
        self._combiner_type = value

    @property
    def scaler_obj(self):
        return self._scaler_obj

    @scaler_obj.setter
    def scaler_obj(self, value):
        if value is not None:
            if not isinstance(value, NumericScaler):
                raise ValueError("Scaler object must be an instance of NumericScaler")
        self._scaler_obj = value

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        if value is None:
            raise ValueError("Columns cannot be None")
        elif type(value) != list:
            raise ValueError("Columns must be a list")
        elif not all(isinstance(x, str) for x in value):
            raise ValueError("Columns must be a list of strings")
        elif len(value) < 2:
            raise ValueError("Columns length must be at least 2")

        self._columns = value

    @property
    def new_feature_name(self):
        return self._new_feature_name

    @new_feature_name.setter
    def new_feature_name(self, value):
        if value is None:
            raise ValueError("New feature name cannot be None")
        elif type(value) != str:
            raise ValueError("New feature name must be a string")
        self._new_feature_name = value

    def transform(self, X):
        match self.combiner_type:
            case "add":
                new_feature = X[self.columns[0]]
                for column in self.columns[1:]:
                    new_feature += X[column]
            case "subtract":
                new_feature = X[self.columns[0]]
                for column in self.columns[1:]:
                    new_feature -= X[column]
            case "multiply":
                new_feature = X[self.columns[0]]
                for column in self.columns[1:]:
                    new_feature *= X[column]
            case "divide":
                new_x = X
                new_feature = new_x[self.columns[0]]
                other_feature = new_x[self.columns[1]].replace(0, np.nan)
                new_feature = new_feature / other_feature

                new_feature = new_feature.replace(np.nan, 0)

            case _:
                raise ValueError("Invalid combiner type")

        new_feature = new_feature.rename(self.new_feature_name)
        if self.scaler_obj is not None:
            new_feature = self.scaler_obj.fit_transform(new_feature)
        return X.with_columns(new_feature)

    def fit(self, X, y=None):
        return self


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper class for a machine learning model that integrates a preprocessing pipeline.
    Parameters
    ----------
    model : object
        The machine learning model to be wrapped.
    prep_pipeline : object
        The preprocessing pipeline to be applied to the data before fitting or predicting.
    Methods
    -------
    fit(X, y=None)
        Fits the model to the data after applying the preprocessing pipeline.
    predict(X)
        Predicts the target values for the given data after applying the preprocessing pipeline.
    transform(X)
        Applies the preprocessing pipeline to the data.
    fit_transform(X, y=None)
        Fits the model to the data after applying the preprocessing pipeline and returns the transformed data.
    Properties
    ----------
    classes_ : array-like
        The classes labels of the model.
    """

    def __init__(self, model, prep_pipeline, **kwargs):
        self.model = model
        self.prep_pipeline = prep_pipeline
        self.kwargs = kwargs

    def fit(self, X, y=None):
        X = self.transform(X, y)

        return self.model.fit(X, y, **self.kwargs)

    def predict(self, X):
        if self.prep_pipeline is not None:
            X = self.prep_pipeline[:-1].transform(X)
            X = self.confirm_columns(X, columns_to_check=self.prep_columns)
            X = self.prep_pipeline[-1].transform(X)
            X = pl.DataFrame(data=X, schema=self.fitted_columns)

        return self.model.predict(X)

    def transform(self, X, y=None):
        if self.prep_pipeline is not None:
            X = self.prep_pipeline[:-1].transform(X)
            self.prep_columns = X.columns
            if y is not None:
                X = self.prep_pipeline[-1].fit_transform(X, y)
                self.fitted_columns = list(
                    self.prep_pipeline[-1].get_feature_names_out()
                )
                X = pl.DataFrame(data=X, schema=self.fitted_columns)
            return X
        else:
            return X

    def fit_transform(self, X, y=None):
        X = self.transform(X, y)
        self.model.fit(X, y, **self.kwargs)
        return X

    @property
    def classes_(self):
        return self.model.classes_

    def confirm_columns(self, X: pl.DataFrame, columns_to_check: list):
        for column in columns_to_check:
            if column not in X.columns:
                new_column = pl.Series(column, [0] * X.shape[0], dtype=pl.Int8)
                X = X.insert_column(columns_to_check.index(column), new_column)
        if X.shape[1] > len(columns_to_check):
            X = X.select(columns_to_check)
        return X

    def predict_proba(self, X):
        if self.prep_pipeline is not None:
            X = self.prep_pipeline[:-1].transform(X)
            X = self.confirm_columns(X, columns_to_check=self.prep_columns)
            X = self.prep_pipeline[-1].transform(X)
            X = pl.DataFrame(data=X, schema=self.fitted_columns)

        return self.model.predict_proba(X)
