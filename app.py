#This imports necessary modules from Flask for 
#building web applications.
from flask import Flask, render_template, request
#is a library used for numerical computations in Python.
import numpy as np
# Pandas is a library used for data manipulation and analysis.
import pandas as pd
#This function splits the dataset into training and 
#testing sets.
from sklearn.model_selection import train_test_split
#This class standardizes features by removing the 
#mean and scaling to unit variance.
from sklearn.preprocessing import StandardScaler
#This class implements logistic regression, a 
#classification algorithm.
from sklearn.linear_model import LogisticRegression
#This class implements a random forest 
#classifier, an ensemble learning method.
from sklearn.ensemble import RandomForestClassifier
#This function computes the accuracy of a classification 
#model.
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__, template_folder=os.getcwd())

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
names = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
         "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
         "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
         "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
         "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
         "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
         "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
try:
    data = pd.read_csv(url, names=names, header=None)
except Exception as e:
    print("Error loading dataset:", e)
    print("Make sure you have an active internet connection.")
    print("Alternatively, you can download the dataset manually and place it in the same directory as this script.")
    exit()

# Drop the 'id' column as it's not useful for classification
data.drop("id", axis=1, inplace=True)

# Convert diagnosis to binary (1 for malignant, 0 for benign)
#This maps the diagnosis column from 
#categorical values ('M' for malignant, 'B' for benign) to numerical values (1 for malignant, 0 for benign).
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Split data into features and target
X = data.drop('diagnosis', axis=1)#This creates the feature matrix X by dropping the diagnosis column.

y = data['diagnosis']#This creates the target vector y containing the diagnosis labels.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
#Standardization is a preprocessing technique used to transform features (variables) so that they have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
log_reg_model = LogisticRegression()#This initializes a logistic regression model.

log_reg_model.fit(X_train_scaled, y_train)#This trains the logistic regression model on the training 
#data.

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)#This initializes a random 
#forest classifier model.
rf_model.fit(X_train_scaled, y_train)#This trains the random forest classifier model on the training 
#data.

# Evaluate models
y_pred_log_reg = log_reg_model.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)##This computes the accuracy of the logistic 
#regression model on the testing data.
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)#This computes the accuracy of the random forest 
#classifier model on the testing data.

# Print testing accuracies
print("Logistic Regression Testing Accuracy:", accuracy_log_reg)
print("Random Forest Testing Accuracy:", accuracy_rf)

# Determine which model to use based on testing accuracy
best_model = log_reg_model if accuracy_log_reg > accuracy_rf else rf_model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        # Get the input features from the form
        features = [float(request.form[feature]) for feature in X.columns]

        # Make prediction
        prediction = best_model.predict([features])[0]

        return render_template("result.html", diagnosis="Malignant" if prediction == 1 else "Benign")

if __name__ == "__main__":
    app.run(debug=True)
