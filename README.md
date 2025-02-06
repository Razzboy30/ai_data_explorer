# ML EDA and Prediction Flask App with Gemini API

This project is a Flask-based web application that allows you to:
- **Upload a CSV dataset**
- **Perform Exploratory Data Analysis (EDA)**
- **Train machine learning models** (Regression or Classification) with automated feature selection using the Gemini API
- **Handle missing values** by dropping features with >50% missing values and imputing others
- **Make predictions** using the trained model via a web interface

## Features

- **CSV File Upload:**  
  Upload your dataset in CSV format through a simple web interface.

- **Exploratory Data Analysis (EDA):**  
  Generate a comprehensive EDA report with:
  - Dataset overview (shape, first few rows, basic statistics)
  - Missing values heatmap
  - Histograms with KDE overlays for numeric features
  - Boxplots for numeric features
  - Correlation heatmap for numeric features
  - Count plots for categorical features
  - Pairplot for feature relationships (if applicable)

- **Feature Selection with Gemini API:**  
  Automatically select the most relevant features for the model. The application sanitizes column names (e.g., renaming `"Sex"` to `"Gender_Encoded"`) to bypass the Gemini API safety filters.

- **Missing Value Handling:**  
  For each feature:
  - **Drop** the feature if more than 50% of the values are missing.
  - **Fill** missing numeric values with the column mean.
  - **Fill** missing categorical values with the column mode.

- **Model Training & Prediction:**  
  Train a model (Linear Regression for regression tasks or RandomForestClassifier for classification tasks) and make predictions based on user input.

- **User-Friendly Interface:**  
  - The **Perform EDA** link opens the analysis report in a new tab.
  - Dynamic forms are used for loading features and making predictions.

## Prerequisites

- Python 3.x
- [Flask](https://flask.palletsprojects.com/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Requests](https://docs.python-requests.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

You can install the required Python packages using pip:

```bash
pip install flask pandas scikit-learn requests matplotlib seaborn
