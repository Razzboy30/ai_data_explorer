import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

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

# Perform EDA & Generate Visualizations
@app.route("/eda/<filename>")
def eda(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(file_path)

    # Store EDA outputs
    eda_results = {
        "head": df.head().to_html(),
        "describe": df.describe().to_html(),
        "unique_values": df.nunique().to_frame().to_html(),
        "shape": f"Rows: {df.shape[0]}, Columns: {df.shape[1]}",
        "missing_values": df.isnull().sum().to_frame().to_html(),
    }

    # Save Plot Helper Function
    def save_plot(fig, filename):
        path = os.path.join("static", filename)
        fig.savefig(path)
        plt.close(fig)
        return filename

    # Missing Values Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")
    eda_results["missing_values_heatmap"] = save_plot(fig, "missing_heatmap.png")

    # Histograms
    fig, ax = plt.subplots(figsize=(15, 10))
    df.hist(ax=ax, bins=30)
    plt.tight_layout()
    eda_results["histograms"] = save_plot(fig, "histograms.png")

    # Boxplot for Outliers
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, orient="h", ax=ax)
    ax.set_title("Boxplot of Numerical Features")
    eda_results["boxplot"] = save_plot(fig, "boxplot.png")

    # Countplot for Categorical Features
    for col in df.select_dtypes(include=["object", "category"]).columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Countplot for {col}")
        save_plot(fig, f"countplot_{col}.png")

    # Pairplot (Slow for large datasets)
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 1:
        pairplot = sns.pairplot(df[num_cols])
        pairplot.savefig(os.path.join("static", "pairplot.png"))
        plt.close()
        eda_results["pairplot"] = "pairplot.png"

    return render_template("eda.html", eda_results=eda_results, filename=filename)

# Train ML Model & Save It
@app.route("/train", methods=["POST"])
def train_model():
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], request.form["filename"])
    df = pd.read_csv(file_path)

    target = request.form["target"]
    model_type = request.form["model"]

    if target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "regression":
        model = LinearRegression()
    elif model_type == "classification":
        model = RandomForestClassifier(n_estimators=100)
    else:
        return jsonify({"error": "Invalid model type"}), 400

    # Train model
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    # Save Model
    model_filename = os.path.join(MODEL_FOLDER, f"{request.form['filename'].split('.')[0]}_{model_type}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    return jsonify({"message": "Model trained successfully!", "accuracy": round(score * 100, 2), "model_file": model_filename})

# Predict using Saved Model
@app.route("/predict", methods=["POST"])
def predict():
    model_filename = request.form["model_file"]

    # Load saved model
    if not os.path.exists(model_filename):
        return jsonify({"error": "Model not found!"}), 400

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    # Convert input data into DataFrame
    input_data = {key: float(value) for key, value in request.form.items() if key != "model_file"}
    df_input = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(df_input)[0]

    return jsonify({"prediction": prediction})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
