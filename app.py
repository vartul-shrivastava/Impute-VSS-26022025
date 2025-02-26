import os
# Removed deep learning related libraries: tensorflow and keras.
from sklearn.ensemble import RandomForestRegressor
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.preprocessing import MinMaxScaler
import io
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import base64
import re
import subprocess 
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, make_response, session
from scipy.stats import gaussian_kde, skew, kurtosis, ks_2samp
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder

from bs4 import BeautifulSoup
import psutil
# -------------------------------------------------
# Flask Setup and Directories
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "your_default_secret_key"  # Replace with a secure key

UPLOAD_FOLDER = "uploads"
PIPELINE_SAVE_DIR = "pipeline_data"
PROJECTS_SAVE_DIR = "projects"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PIPELINE_SAVE_DIR, exist_ok=True)
os.makedirs(PROJECTS_SAVE_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/system_stats', methods=['GET'])
def system_stats():
    cpu_utilization = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_total = ram_info.total / (1024 ** 3)
    ram_available = ram_info.available / (1024 ** 3)
    ram_utilization = ram_info.percent
    stats = {
        "cpu_utilization_percent": cpu_utilization,
        "ram_utilization_percent": ram_utilization,
    }
    return jsonify(stats), 200

# -------------------------------------------------
# Global In-Memory Data Store
# -------------------------------------------------
data_store = {
    "original_data": None,        # The active/working DataFrame
    "imputed_versions": {},       # { column: { method: pd.Series } } (temporary imputed copies)
    "imputed_stats": {},          # { column: { method: stats_dict } }
    "pipeline_steps": []          # Pipeline steps (if needed)
}

# Note: Deep learning–based imputation functions (e.g. SDAE, MIDA) have been removed.

# -------------------------------------------------
# Helper Functions (Imputation, KDE, Stats, etc.)
# -------------------------------------------------
def compute_kde(values):
    if len(values) < 2:
        return [], []
    kde_func = gaussian_kde(values)
    kde_x = np.linspace(min(values), max(values), 100)
    kde_y = kde_func(kde_x)
    return kde_x.tolist(), kde_y.tolist()

def compute_comparative_stats(original_vals, imputed_vals, orig_kde, imp_kde):
    # Compute statistics for numeric columns only.
    def safe_stats(arr):
        from numpy import mean, median, std
        if len(arr) == 0:
            return None, None, None, None, None
        return (float(mean(arr)), float(median(arr)), float(std(arr)), float(skew(arr)), float(kurtosis(arr)))
    orig_mean, orig_med, orig_std, orig_skew, orig_kurt = safe_stats(original_vals)
    imp_mean, imp_med, imp_std, imp_skew, imp_kurt = safe_stats(imputed_vals)
    x_orig, y_orig = orig_kde
    x_imp, y_imp = imp_kde
    overlap_value = None
    kl_divergence = None
    if len(x_orig) == len(x_imp) and len(x_orig) > 1:
        overlap_value = float(np.trapz(np.minimum(y_orig, y_imp), x_orig))
        y_orig_norm = y_orig / np.trapz(y_orig, x_orig)
        y_imp_norm = y_imp / np.trapz(y_imp, x_orig)
        kl_divergence = float(np.sum(y_orig_norm * np.log(y_orig_norm / y_imp_norm)))
    ks_stat, ks_pvalue = None, None
    if len(original_vals) > 1 and len(imputed_vals) > 1:
        stat_val, p_val = ks_2samp(original_vals, imputed_vals)
        ks_stat, ks_pvalue = float(stat_val), float(p_val)
    return {
        "original_mean": orig_mean,
        "imputed_mean": imp_mean,
        "original_median": orig_med,
        "imputed_median": imp_med,
        "original_std": orig_std,
        "imputed_std": imp_std,
        "kde_overlap": overlap_value,
        "original_skew": orig_skew,
        "imputed_skew": imp_skew,
        "original_kurtosis": orig_kurt,
        "imputed_kurtosis": imp_kurt,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "kl_divergence": kl_divergence
    }

def compute_categorical_stats(original_vals, imputed_vals):
    """Compute categorical statistics: mode, unique count, mode frequency, and unique percentage."""
    def get_stats(vals):
        series = pd.Series(vals)
        total = len(series)
        if series.empty or total == 0:
            return {"mode": None, "unique_count": 0, "mode_frequency": 0, "unique_percentage": 0}
        mode_val = series.mode().iloc[0] if not series.mode().empty else None
        unique_count = series.nunique()
        mode_freq = int((series == mode_val).sum()) if mode_val is not None else 0
        unique_percentage = (unique_count / total) * 100 if total > 0 else 0
        return {
            "mode": mode_val,
            "unique_count": unique_count,
            "mode_frequency": mode_freq,
            "unique_percentage": unique_percentage
        }

    orig_stats = get_stats(original_vals)
    imp_stats = get_stats(imputed_vals)
    return {"original": orig_stats, "imputed": imp_stats}

def encode_predictor_features(df, features):
    """
    Given a DataFrame and a list of feature column names,
    encode any non-numeric columns using LabelEncoder.
    Returns the transformed DataFrame and a dictionary mapping column names to their encoders.
    """
    df_encoded = df.copy()
    encoders = {}
    for col in features:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    return df_encoded, encoders

def apply_imputation(column_name, method, constant_value):
    df = data_store["original_data"].copy()
    is_numeric = pd.api.types.is_numeric_dtype(df[column_name])
    # Supported methods for numeric columns.
    numeric_methods = ["mean", "median", "rf", "svr", "gb"]
    if not is_numeric and method in numeric_methods:
        raise ValueError(f"Method '{method}' is only for numeric columns, but '{column_name}' is categorical.")

    # Simple imputation methods for a single column:
    if method == "mean":
        return df[column_name].fillna(df[column_name].mean())
    elif method == "median":
        return df[column_name].fillna(df[column_name].median())
    elif method == "mode":
        mode_val = df[column_name].mode(dropna=True)
        if len(mode_val) == 0:
            raise ValueError(f"No valid mode found for '{column_name}'.")
        return df[column_name].fillna(mode_val[0])
    elif method == "constant":
        if constant_value is None or str(constant_value).strip() == "":
            raise ValueError("A constant value must be provided for constant imputation.")
        if is_numeric:
            try:
                constant_numeric = float(constant_value)
                return df[column_name].fillna(constant_numeric)
            except ValueError:
                raise ValueError("A numeric constant value must be provided for numeric columns.")
        else:
            return df[column_name].fillna(constant_value)

    elif method in ["rf", "svr", "gb"]:
        # Define predictors as all columns except the target.
        features = df.columns.drop(column_name)
        # Encode categorical predictor columns.
        df_encoded, encoders = encode_predictor_features(df, features)
        
        non_missing = df_encoded[df_encoded[column_name].notna()]
        missing = df_encoded[df_encoded[column_name].isna()]
        if missing.empty:
            return df[column_name]
        # Ensure that predictor features in the training set have no missing values.
        non_missing = non_missing.dropna(subset=features)
        if non_missing.empty:
            raise ValueError(f"Not enough data to train {method} imputation.")
        X_train = non_missing[features]
        y_train = non_missing[column_name]

        if method == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        elif method == "svr":
            model = SVR()
        elif method == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        
        X_missing = missing[features].copy()
        # For any remaining missing predictor values, fill with the column mean from the encoded data.
        for col in features:
            X_missing[col].fillna(df_encoded[col].mean(), inplace=True)
        y_pred = model.predict(X_missing)
        # Assign predictions to the original dataframe.
        df.loc[missing.index, column_name] = y_pred
        return df[column_name]

    elif method in ["knn", "mice"]:
        imputed_df = impute_with_sklearn_dataset_with_labelencoder(df, method)
        return imputed_df[column_name]
    elif method == "complete-case":
        return df[column_name].dropna()
    elif method == "random":
        non_missing = df[column_name].dropna()
        if non_missing.empty:
            raise ValueError(f"No non-missing values available in column '{column_name}' for random imputation.")
        return df[column_name].apply(lambda x: x if pd.notna(x) else non_missing.sample(1).iloc[0])
    else:
        raise ValueError(f"Unknown imputation method: {method}")

def impute_with_sklearn(df, column_name, method, is_numeric):
    """
    Impute a single column using sklearn's imputation methods, but using the entire dataset 
    as context. The function first converts categorical variables to numeric using LabelEncoder,
    applies the imputer (KNN or MICE) to the full dataset, and then decodes any encoded columns.
    
    Parameters:
      - df: The DataFrame containing the data.
      - column_name: The column to be imputed.
      - method: The imputation method ('knn' or 'mice').
      - is_numeric: Boolean indicating if the column is numeric.
      
    Returns:
      - A pandas Series with the imputed values for column_name.
    """
    # Create a copy of the dataframe to impute on
    impute_df = df.copy()

    # Identify categorical columns that need encoding (including the target column if not numeric)
    categorical_cols = impute_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Prepare a dictionary to store the encoders
    label_encoders = {}

    # Encode the categorical columns using LabelEncoder.
    for col in categorical_cols:
        le = LabelEncoder()
        impute_df[col] = le.fit_transform(impute_df[col].astype(str))
        label_encoders[col] = le

    # For the target column, if it's non-numeric but marked as numeric by is_numeric=False,
    # force conversion to numeric (with errors coerced) so that the imputer works.
    if not is_numeric and column_name not in categorical_cols:
        impute_df[column_name] = pd.to_numeric(impute_df[column_name], errors='coerce')

    # Select the imputer
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f"Method '{method}' not supported in impute_with_sklearn.")

    # Apply the imputer to the entire DataFrame.
    imputed_array = imputer.fit_transform(impute_df)
    imputed_df = pd.DataFrame(imputed_array, columns=impute_df.columns, index=impute_df.index)

    # For each originally categorical column, round and decode back to strings.
    for col in categorical_cols:
        le = label_encoders[col]
        imputed_df[col] = imputed_df[col].round().astype(int)
        imputed_df[col] = le.inverse_transform(imputed_df[col])
    
    return imputed_df[column_name]

mvgen_data = None  # Global storage for MVGenerator dataset

@app.route("/mvgen-upload", methods=["POST"])
def mvgen_upload():
    """
    Uploads a CSV file for the MVGenerator, stores it in mvgen_data,
    and returns a missing matrix along with column and row information.
    """
    global mvgen_data
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a CSV file."}), 400
    try:
        df = pd.read_csv(file)
        mvgen_data = df  # store globally
        missing_matrix = df.isna().astype(int).values.tolist()
        columns = df.columns.tolist()
        rows = df.index.tolist()
        return jsonify({
            "success": True,
            "missing_matrix": missing_matrix,
            "columns": columns,
            "rows": rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/mvgen-generate", methods=["POST"])
def mvgen_generate():
    """
    Induces missing values on a specified column.
    Request JSON must contain:
      - "column": column name,
      - "percentage": percentage of rows to set as missing,
      - "method": "random" or "uniform".
    Returns an updated missing matrix.
    """
    global mvgen_data
    if mvgen_data is None:
        return jsonify({"error": "No dataset loaded for MVGenerator."}), 400
    req = request.json
    col = req.get("column")
    try:
        percentage = float(req.get("percentage"))
    except Exception:
        return jsonify({"error": "Invalid percentage value."}), 400
    method = req.get("method")
    if col not in mvgen_data.columns:
        return jsonify({"error": f"Column '{col}' not found."}), 400
    total_rows = len(mvgen_data)
    num_to_remove = int((percentage / 100.0) * total_rows)
    if num_to_remove <= 0:
        return jsonify({"error": "No rows to remove. Increase the percentage."}), 400

    try:
        indices = np.random.choice(mvgen_data.index, size=num_to_remove, replace=False)
        mvgen_data.loc[indices, col] = np.nan
    except:
        return jsonify({"Some unexpected error has come"}), 400

    missing_matrix = mvgen_data.isna().astype(int).values.tolist()
    return jsonify({
        "success": True,
        "message": f"Missing values induced in column '{col}' by {percentage}% using '{method}' method.",
        "missing_matrix": missing_matrix
    })

@app.route("/mvgen-download", methods=["GET"])
def mvgen_download():
    """
    Returns the modified MVGenerator dataset as a CSV file.
    """
    global mvgen_data
    if mvgen_data is None:
        return jsonify({"error": "No dataset available."}), 400
    csv_str = mvgen_data.to_csv(index=False)
    response = make_response(csv_str)
    response.headers["Content-Disposition"] = "attachment; filename=mvgen_modified.csv"
    response.mimetype = "text/csv"
    return response

@app.route("/get-current-dataset", methods=["GET"])
def get_current_dataset():
    """
    Returns the current snapshot of the dataset under imputation as a list of dictionaries.
    Replaces any NaN/Inf values with None so the JSON is valid.
    """
    try:
        df = data_store.get("original_data")
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400

        # Replace NaN and infinity with None so we can return valid JSON
        df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})

        # Convert to a list of dictionaries
        dataset_records = df_clean.to_dict(orient="records")

        # Return JSON response
        return jsonify({
            "success": True,
            "dataset": dataset_records,
            "numRows": len(dataset_records)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def impute_with_sklearn_dataset_with_labelencoder(df, method):
    """
    Impute the entire dataset using sklearn's imputation methods.
    This function encodes all columns so that every column is numeric.
    Then, the chosen imputer (KNN or MICE) is applied to the entire DataFrame.
    Finally, the originally categorical columns are decoded back to their original labels.
    """
    # Create a copy of the entire dataset.
    impute_df = df.copy()
    
    # Identify all columns and determine which are categorical.
    categorical_cols = impute_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Dictionary to hold label encoders.
    label_encoders = {}
    
    # Encode each categorical column.
    for col in categorical_cols:
        le = LabelEncoder()
        impute_df[col] = le.fit_transform(impute_df[col].astype(str))
        label_encoders[col] = le
    
    # Select the imputer.
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    else:
        raise ValueError(f"Method '{method}' not supported for dataset-wide imputation.")
    
    # Fit and transform the entire DataFrame.
    imputed_array = imputer.fit_transform(impute_df)
    imputed_df = pd.DataFrame(imputed_array, columns=impute_df.columns, index=impute_df.index)
    
    # For each originally categorical column, round and decode back to strings.
    for col in categorical_cols:
        le = label_encoders[col]
        imputed_df[col] = imputed_df[col].round().astype(int)
        imputed_df[col] = le.inverse_transform(imputed_df[col])
    
    return imputed_df

def update_dataset_with_imputation(col, method, constant_value):
    # For constant imputation, format key as "constant (value)"
    key = method if method != "constant" else f"constant ({constant_value})"
    if method != "complete-case":
        imputed_series = data_store["imputed_versions"][col][key]
        data_store["original_data"][col] = imputed_series
    missing_matrix = data_store["original_data"].isna().astype(int).values.tolist()
    return missing_matrix

# -------------------------------------------------
# Route: Push a Chosen Imputed Column to Main DataFrame
# -------------------------------------------------
@app.route("/push-imputation", methods=["POST"], endpoint="push_imputation_route")
def push_imputation_route():
    try:
        req = request.json
        col = req.get("column")
        method = req.get("method")
        constant_value = req.get("constant_value")  # For constant imputation
        if not col or not method:
            return jsonify({"error": "Both column and method must be specified."}), 400

        if method in ["knn", "mice"]:
            # For KNN/MICE, impute the entire dataset.
            imputed_df = impute_with_sklearn_dataset_with_labelencoder(data_store["original_data"], method)
            data_store["original_data"].update(imputed_df)
            missing_matrix = data_store["original_data"].isna().astype(int).values.tolist()
            return jsonify({
                "message": f"Dataset imputed using '{method}' method.",
                "updated_matrix": missing_matrix
            })
        else:
            # For constant and column-specific methods.
            if method == "constant":
                if constant_value is None or str(constant_value).strip() == "":
                    return jsonify({"error": "A constant value must be provided for constant imputation."}), 400
                matrix = update_dataset_with_imputation(col, method, constant_value)
            else:
                matrix = update_dataset_with_imputation(col, method, None)
            return jsonify({
                "message": f"Imputation '{method}' for column '{col}' pushed to main dataset.",
                "updated_matrix": matrix
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Route: Impute Entire Dataset
# -------------------------------------------------
@app.route("/impute-dataset", methods=["POST"])
def impute_dataset():
    try:
        req = request.json
        method = req.get("method")
        cval = req.get("constant_value")
        if not method:
            return jsonify({"error": "No imputation method provided."}), 400
        df = data_store["original_data"].copy()
        if method in ["knn", "mice"]:
            # Use the dataset-wide imputation with label encoding.
            imputed_df = impute_with_sklearn_dataset_with_labelencoder(df, method)
            data_store["original_data"] = imputed_df.copy()
        else:
            # Otherwise, process each column individually.
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    try:
                        imputed_col = apply_imputation(col, method, cval)
                        df[col] = imputed_col
                    except Exception as ex:
                        print(f"Skipping imputation for column {col} due to error: {ex}")
            data_store["original_data"] = df.copy()
        matrix = data_store["original_data"].isna().astype(int).values.tolist()
        return jsonify({
            "message": f"Dataset imputed using method '{method}'.",
            "updated_matrix": matrix
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Existing Routes: Uploading and Column Data
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process-dataset", methods=["POST"])
def process_dataset_route():
    try:
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are allowed."}), 400
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        df = pd.read_csv(path)
        os.remove(path)
        data_store["original_data"] = df
        data_store["imputed_versions"] = {}
        data_store["imputed_stats"] = {}
        data_store["pipeline_steps"] = []
        missing_matrix = df.isna().astype(int).values.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
        rows = df.index.tolist()
        return jsonify({
            "matrix": missing_matrix,
            "columns": df.columns.tolist(),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "rows": rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/column-data", methods=["POST"])
def column_data():
    try:
        req = request.json
        col = req.get("column")
        if not col:
            return jsonify({"error": "No column provided."}), 400
        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found in dataset."}), 400
        col_data = df[col]
        distribution_values = col_data.dropna().tolist()
        missing_values = col_data.isna().astype(int).tolist()
        kde_x, kde_y = [], []
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        if is_numeric:
            kde_x, kde_y = compute_kde(distribution_values)
        return jsonify({
            "distribution_values": distribution_values,
            "missing_values": missing_values,
            "kde_x": kde_x,
            "kde_y": kde_y,
            "is_numeric": is_numeric
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import time

@app.route("/impute", methods=["POST"])
def impute_route():
    try:
        data = request.json
        col = data.get("column")
        method = data.get("method")
        cval = data.get("constant_value")
        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not found."}), 400
        original_col = df[col].copy()
        if original_col.isna().sum() == 0 and method != "complete-case":
            return jsonify({"error": f"No missing values in column '{col}'."}), 400

        # Start timer before imputation
        start_time = time.time()
        imputed_col = apply_imputation(col, method, cval)
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        if col not in data_store["imputed_versions"]:
            data_store["imputed_versions"][col] = {}
        key = method if method != "constant" else f"{method} ({cval})"
        data_store["imputed_versions"][col][key] = imputed_col
        data_store["pipeline_steps"].append({
            "column": col,
            "method": method,
            "config": {"value": cval},
            "time_taken": elapsed_time  # Store time taken for this step
        })
        orig_vals = original_col.dropna().tolist()
        imp_vals = imputed_col.tolist()
        is_numeric = pd.api.types.is_numeric_dtype(original_col)
        if is_numeric:
            kde_x_orig, kde_y_orig = compute_kde(orig_vals)
            kde_x_imp, kde_y_imp = compute_kde(imp_vals)
            stats = compute_comparative_stats(
                orig_vals, imp_vals, (kde_x_orig, kde_y_orig), (kde_x_imp, kde_y_imp)
            )
        else:
            stats = compute_categorical_stats(orig_vals, imp_vals)
        if method == "constant":
            stats["constant_value"] = cval
        
        # Add time taken into the stats
        stats["time_taken"] = elapsed_time

        if col not in data_store["imputed_stats"]:
            data_store["imputed_stats"][col] = {}
        data_store["imputed_stats"][col][key] = stats
        matrix = data_store["original_data"].isna().astype(int).values.tolist()
        return jsonify({
            "message": f"Imputation '{method}' applied to '{col}'.",
            "original_distribution": orig_vals,
            "imputed_distribution": imp_vals,
            "kde_x_original": kde_x_orig if is_numeric else [],
            "kde_y_original": kde_y_orig if is_numeric else [],
            "kde_x_imputed": kde_x_imp if is_numeric else [],
            "kde_y_imputed": kde_y_imp if is_numeric else [],
            "stats": stats,
            "updated_matrix": matrix
        })
    except Exception as e:
        return jsonify({"error": f"Imputation failed: {str(e)}"}), 500


@app.route("/get-stats", methods=["POST"])
def get_stats_route():
    try:
        req = request.json
        col = req.get("column")
        col_stats = data_store["imputed_stats"].get(col, {})
        return jsonify({"stats": col_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-imputations", methods=["POST"])
def get_imputations():
    try:
        req = request.json
        col = req.get("column")
        col_imps = data_store["imputed_versions"].get(col, {})
        response = {}
        for method, imputed_value in col_imps.items():
            if isinstance(imputed_value, list):
                # For each element, if it's a pandas Series, convert to list; if it's already a list, leave it.
                response[method] = [s.tolist() if isinstance(s, pd.Series) else s for s in imputed_value]
            else:
                response[method] = imputed_value.tolist() if isinstance(imputed_value, pd.Series) else imputed_value
        return jsonify({"imputations": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/save-pipeline", methods=["POST"])
def save_pipeline():
    try:
        pipeline_data = request.json.get("pipeline", [])
        if not isinstance(pipeline_data, list):
            return jsonify({"error": "Invalid pipeline format; must be a list."}), 400
        spath = os.path.join(PIPELINE_SAVE_DIR, "pipeline.json")
        with open(spath, "w") as f:
            json.dump(pipeline_data, f, indent=2)
        return jsonify({"success": True, "message": "Pipeline saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-imputed-csv", methods=["GET"])
def download_imputed_csv():
    try:
        if data_store["original_data"] is None:
            return jsonify({"error": "No dataset loaded."}), 400
        df = data_store["original_data"].copy()
        csv_str = df.to_csv(index=False)
        response = make_response(csv_str)
        response.headers["Content-Disposition"] = "attachment; filename=imputed_dataset.csv"
        response.mimetype = "text/csv"
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/calculate-correlation", methods=["POST"])
def calculate_correlation():
    try:
        req = request.json
        column = req.get("column")
        imputed_values = req.get("imputed", None)
        if not column:
            return jsonify({"error": "No column specified."}), 400

        df = data_store["original_data"]
        if df is None:
            return jsonify({"error": "No dataset loaded."}), 400
        if column not in df.columns:
            return jsonify({"error": f"Column '{column}' not found in the dataset."}), 400

        numeric_df = df.select_dtypes(include=[np.number])
        if column not in numeric_df.columns:
            return jsonify({"error": f"Column '{column}' is not numeric."}), 400

        if imputed_values:
            numeric_df[column] = pd.Series(imputed_values)

        correlations = numeric_df.corr()[column].drop(index=column).to_dict()
        response = {
            "columns": list(correlations.keys()),
            "correlations": list(correlations.values()),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Run the App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
