# Code Modularity is Extremely important
# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# pip install sklearn-pandas
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import sklearn.metrics as skmet
import pickle

import warnings

warnings.filterwarnings("ignore")

# Load the Data

cancerdata = pd.read_csv(
    r"C:\Users\gaura\OneDrive\Desktop\Cancer_prediction_end_to_end\Data sets\cancerdata.csv"
)

# MySQL Database connection

from sqlalchemy import create_engine

# Creating engine which connect to MySQL

user = "root"
pw = "Kabali123$"
db = "cancer_db"

# Creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Dumping data into database
cancerdata.to_sql(
    "cancer", con=engine, if_exists="replace", chunksize=1000, index=False
)

# Loading data from database
sql = "select * from cancer"

cancerdf = pd.read_sql(sql, con=engine)

print(cancerdf)

# Data Preprocessing and EDA
# Converting 'B' to Benign and 'M' to Malignant

cancerdf["diagnosis"] = np.where(
    cancerdf["diagnosis"] == "B", "Benign", cancerdf["diagnosis"]
)

cancerdf["diagnosis"] = np.where(
    cancerdf["diagnosis"] == "M", "Malignant", cancerdf["diagnosis"]
)

print(cancerdf)

cancerdf.drop(["id"], axis=1, inplace=True)
cancerdf.info()

cancerdf.describe()

# Seggregating input and output variables
cancerdf_X = pd.DataFrame(cancerdf.iloc[:, 1:])

cancerdf_y = pd.DataFrame(cancerdf.iloc[:, 0])

# EDA and Data Preparation
cancerdf_X.info()

cancerdf_X.isnull().sum()

# All numeric features
numeric_features = cancerdf_X.select_dtypes(exclude=["object"]).columns

numeric_features

# Imputation strategy for numeric columns
num_pipeline = Pipeline([("impute", SimpleImputer(strategy="mean"))])

# All Categorical features
categorical_features = cancerdf_X.select_dtypes(include=["object"]).columns

print(categorical_features)

# DataFrameMapper is used to map the given attribute
# Encoding categorical to numeric variable
categ_pipeline = Pipeline(
    [("label", DataFrameMapper([(categorical_features, OneHotEncoder(drop="first"))]))]
)

# Using ColumnTransformer to transform the columns of an array or Pandas DataFrame
preprocess_pipeline = ColumnTransformer(
    [
        ("categorical", categ_pipeline, categorical_features),
        ("numerical", num_pipeline, numeric_features),
    ]
)

processed = preprocess_pipeline.fit(cancerdf_X)  # fit the pipeline

processed

# Save the defined pipeline
import joblib

joblib.dump(processed, "processed1")

import os

os.getcwd()

# Transform the original data using the pipeline defined above
cancerclean = pd.DataFrame(processed.transform(cancerdf_X), columns=cancerdf_X.columns)

cancerclean.info()
cancerclean.isnull().sum()

# Define scaling pipeline
scale_pipeline = Pipeline([("scale", MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer(
    [("scale", scale_pipeline, cancerclean.columns)]
)

processed2 = preprocess_pipeline2.fit(cancerclean)
processed2

# Save the Scaling Pipeline
joblib.dump(processed2, "processed2")

import os

os.getcwd()

# Normalized Data Frame
cancerclean_n = pd.DataFrame(
    processed2.transform(cancerclean), columns=cancerclean.columns
)
cancerclean_n

eda = cancerclean_n.describe()
eda

# Output variable is stored in a different object
Y = np.array(cancerdf_y["diagnosis"])  # Target

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    cancerclean_n, Y, test_size=0.2, random_state=0
)

print(f"Shape of Train data: ", X_train.shape)
print(f"Shape of Test data: ", X_test.shape)


# Model Building
knn = KNeighborsClassifier(n_neighbors=21)

KNN = knn.fit(X_train, y_train)

# Evaluate the model
pred_train = knn.predict(X_train)
pred_train

# Cross Tables
pd.crosstab(y_train, pred_train, rownames=["Actual"], colnames=["Predictions"])

skmet.accuracy_score(y_train, pred_train)

# Predict the class on test data
pred_test = knn.predict(X_test)
pred_test

# Evaluate the model with test data
skmet.accuracy_score(y_test, pred_test)

pd.crosstab(y_test, pred_test, rownames=["Actual"], colnames=["Predicted"])

cm = skmet.confusion_matrix(y_test, pred_test)
cm

# Creating empty list variable
acc = []

# running KNN algorithm from 3 to 50 nearest neighbors (odd numbers) and stroing the accuracy values

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    diff = train_acc - test_acc
    acc.append([diff, train_acc, test_acc])

acc

# plotting the data frequencies
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "ro-")
plt.plot(np.arange(3, 50, 2), [i[2] for i in acc], "bo-")

# Save the model

knn = KNeighborsClassifier(n_neighbors=9)
KNN = knn.fit(X_train, y_train)

knn_best = KNN
pickle.dump(knn_best, open("knn.pkl", "wb"))

import os

os.getcwd()
