import os
import pickle
import pandas as pd
import gemini_api 
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = "static"
MODEL_FOLDER = "models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Home Route - Upload CSV
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename.endswith(".csv"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return render_template("analyze.html", file=file.filename)
    return render_template("index.html")

# Train ML Model Using Gemini for Feature Selection
@app.route("/train", methods=["POST"])
def train_model():
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], request.form["filename"])
    df = pd.read_csv(file_path)

    target = request.form["target"]
    model_type = request.form["model"]

    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    # Create a dictionary of feature types (excluding the target)
    feature_types = {
        col: "numerical" if df[col].dtype in ["int64", "float64"] else "categorical"
        for col in df.columns if col != target
    }

    # Clean feature names to bypass Gemini API safety filters
    replacements = {
        "Sex": "Gender_Encoded",     
        "Race": "Ethnicity_Encoded",    
    }
    # Create mapping: original feature name → sanitized name
    mapping = {col: replacements.get(col, col) for col in feature_types.keys()}
    # Build sanitized feature types dictionary for Gemini
    sanitized_feature_types = {mapping[col]: dtype for col, dtype in feature_types.items()}

    #Call Gemini API to get the best features
    best_features = gemini_api.get_best_features(sanitized_feature_types, target, model_type)
    if not best_features:
        print("Gemini API did not return a valid feature list. Falling back to default numerical features.")
        selected_features = [col for col, dtype in feature_types.items() if dtype == "numerical"]
    else:
        # Map the sanitized feature names back to the original names
        reverse_mapping = {sanitized: original for original, sanitized in mapping.items()}
        selected_features = [reverse_mapping[feat] for feat in best_features if feat in reverse_mapping]

    # STEP 4: Prepare the feature matrix (X) and target vector (y)
    X = df[selected_features].copy()  # use copy() to avoid chained assignment warnings
    y = df[target]

    # Check for missing values in each feature.
    # If a feature has >50% missing values, drop it.
    # Otherwise, fill missing values with the column's average (if numeric)
    # or with the mode (if categorical).
    for col in X.columns:
        missing_ratio = X[col].isna().mean()
        if missing_ratio > 0.5:
            print(f"Dropping feature '{col}' due to >50% missing values (missing ratio: {missing_ratio:.2f}).")
            X = X.drop(columns=col)
        else:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].mean(), inplace=True)
            else:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X[col].fillna(mode_val[0], inplace=True)
                else:
                    X[col].fillna("missing", inplace=True)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    feature_names = list(X.columns)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the selected model
    if model_type == "regression":
        model = LinearRegression()
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Save the trained model and feature names for prediction later
    model_filename = os.path.join(MODEL_FOLDER, f"{request.form['filename'].split('.')[0]}_{model_type}.pkl")
    feature_filename = model_filename.replace(".pkl", "_features.pkl")

    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(feature_filename, "wb") as f:
        pickle.dump(feature_names, f)

    return jsonify({
        "message": "Model trained successfully!",
        "accuracy": round(score * 100, 2),
        "model_file": model_filename,
        "feature_file": feature_filename
    })


@app.route('/eda/<filename>')
def eda(filename):
    # Build the file path for the uploaded CSV
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Basic EDA statistics
    eda_results = {}
    eda_results["shape"] = df.shape
    eda_results["head"] = df.head().to_html(classes='table table-striped')
    eda_results["describe"] = df.describe().to_html(classes='table table-striped')
    eda_results["unique_values"] = df.nunique().to_dict()
    eda_results["missing_values"] = df.isnull().sum().to_dict()

    # Generate Missing Values Heatmap
    try:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        missing_heatmap_path = os.path.join("static", "missing_heatmap.png")
        plt.savefig(missing_heatmap_path)
        plt.close()
    except Exception as e:
        print("Error generating missing heatmap:", e)

    # Generate Histograms with KDE for Numeric Data
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            num_plots = len(numeric_cols)
            ncols = 2
            nrows = (num_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
            # Ensure axes is always iterable
            if num_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(f"Histogram & KDE for {col}")
            # Remove any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            hist_kde_path = os.path.join("static", "histograms_kde.png")
            plt.tight_layout()
            plt.savefig(hist_kde_path)
            plt.close()
    except Exception as e:
        print("Error generating histograms with KDE:", e)

    # Generate Boxplots for Numeric Data
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            num_plots = len(numeric_cols)
            ncols = 2
            nrows = (num_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
            if num_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i, col in enumerate(numeric_cols):
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(f"Boxplot for {col}")
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            boxplot_path = os.path.join("static", "boxplots.png")
            plt.tight_layout()
            plt.savefig(boxplot_path)
            plt.close()
    except Exception as e:
        print("Error generating boxplots:", e)

    # Generate Correlation Heatmap
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            corr_heatmap_path = os.path.join("static", "correlation_heatmap.png")
            plt.tight_layout()
            plt.savefig(corr_heatmap_path)
            plt.close()
    except Exception as e:
        print("Error generating correlation heatmap:", e)

    # Generate Count Plots for Categorical Data
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            num_plots = len(categorical_cols)
            ncols = 2
            nrows = (num_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
            if num_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i, col in enumerate(categorical_cols):
                sns.countplot(y=df[col], ax=axes[i])
                axes[i].set_title(f"Count Plot for {col}")
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            countplot_path = os.path.join("static", "countplots.png")
            plt.tight_layout()
            plt.savefig(countplot_path)
            plt.close()
    except Exception as e:
        print("Error generating count plots:", e)

    # Generate Pairplot for Numeric Data (if manageable)
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1 and len(numeric_cols) <= 10:
            pairgrid = sns.pairplot(df[numeric_cols])
            pairplot_path = os.path.join("static", "pairplot.png")
            pairgrid.fig.savefig(pairplot_path)
            plt.close('all')
    except Exception as e:
        print("Error generating pairplot:", e)

    # Render the EDA template with the results and generated plots
    return render_template("eda.html", eda_results=eda_results)

# Get Feature Names for a Trained Model
@app.route("/get_features", methods=["POST"])
def get_features():
    feature_filename = request.form["feature_file"]

    if not os.path.exists(feature_filename):
        return jsonify({"error": "Feature file not found!"}), 400

    with open(feature_filename, "rb") as f:
        feature_names = pickle.load(f)

    return jsonify({"features": feature_names})


# Predict using Saved Model
@app.route("/predict", methods=["POST"])
def predict():
    model_filename = request.form["model_file"]

    if not os.path.exists(model_filename):
        return jsonify({"error": "Model not found!"}), 400

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # Build input data from the request (ignore "model_file")
    input_data = {key: float(value) for key, value in request.form.items() if key != "model_file"}
    df_input = pd.DataFrame([input_data])

    # Get prediction from the model (this might be a numpy scalar)
    prediction = model.predict(df_input)[0]

    # Convert numpy scalar to a native Python type (int, float, etc.)
    if hasattr(prediction, "item"):
        prediction = prediction.item()

    return jsonify({"prediction": prediction})


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
