from flask import Flask, request, render_template, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 📌 Homepage Route (Upload CSV)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename.endswith(".csv"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return render_template("analyze.html", file=file.filename)
    return render_template("index.html")

# 📌 Load CSV & Perform EDA
@app.route("/eda/<filename>")
def eda(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    df = pd.read_csv(file_path)

    # Quick Stats
    stats = df.describe().to_html()

    # Correlation Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    heatmap = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("eda.html", stats=stats, heatmap=heatmap, filename=filename)

# 📌 Train ML Model (Regression/Classification)
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

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return jsonify({"message": "Model trained successfully!", "accuracy": round(score * 100, 2)})

if __name__ == "__main__":
    app.run(debug=True)
