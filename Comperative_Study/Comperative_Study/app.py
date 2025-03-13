import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import os
import time

# Flask app setup
app = Flask(__name__, template_folder="Templates")
UPLOAD_FOLDER = "Uploads"
STATIC_FOLDER = "Static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global results dictionary
results = {}

# Define a dictionary of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part provided", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if not file.filename.endswith('.csv'):
        return "Only CSV files are allowed", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
    except Exception as e:
        return f"Error processing file: {e}", 500

    return render_template('select_columns.html', file_path=file_path, columns=columns)

@app.route('/process', methods=['POST'])
def process_file():
    global results
    file_path = request.form.get('file_path')
    input_columns = request.form.get('input_columns')
    output_column = request.form.get('output_column')

    # Ensure input_columns is a proper list
    if input_columns:
        input_columns = [col.strip() for col in input_columns.split(',')]  

    if not file_path or not input_columns or not output_column:
        return "Invalid input or missing columns", 400

    try:
        data = pd.read_csv(file_path)
        X = data[input_columns]
        y = data[output_column]
    except Exception as e:
        return f"Error processing data: {e}", 500

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results.clear()  # Clear previous results

    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
            results[name] = scores.mean()
        except Exception as e:
            results[name] = f"Error: {e}"

    # Save the bar chart as an image with cache-busting
    timestamp = int(time.time())
    chart_filename = f"accuracy_chart_{timestamp}.png"
    chart_path = os.path.join(STATIC_FOLDER, chart_filename)

    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.xlabel('Algorithm')
    plt.ylabel('Average Accuracy')
    plt.title('Comparative Study of Machine Learning Algorithms with K-Fold Cross-Validation')
    plt.xticks(rotation=45)
    plt.savefig(chart_path)
    plt.close()

    # Identify the best algorithm
    valid_results = {k: v for k, v in results.items() if isinstance(v, (float, int))}
    best_algo = max(valid_results, key=valid_results.get) if valid_results else "No valid results"

    return render_template('results.html', best_algo=best_algo, accuracy=results.get(best_algo, "N/A"), chart_path=chart_filename, results=results)

@app.route('/api/results')
def api_results():
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
