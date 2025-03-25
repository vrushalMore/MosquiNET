import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data():
    df = pd.read_csv("../../data/raw/data.csv")
    X = df.drop(columns=['Habitable', 'Area Name'])
    y = df['Habitable']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Naive Bayes Model:")
    print("  Accuracy:", accuracy_score(y_test, y_pred))
    print("  Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("  Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("  F1 Score:", f1)

    return f1, model

X_train, X_test, y_train, y_test = load_data()
X_train, X_test = preprocess(X_train, X_test)

best_f1 = 0.0
best_model = None

for i in range(3):  
    model = train_naive_bayes(X_train, y_train)
    f1, trained_model = evaluate(model, X_test, y_test)

    if f1 > best_f1:
        best_f1 = f1
        best_model = trained_model

if best_model:
    with open("naiveBayesModel.pkl", "wb") as f:
        pickle.dump(best_model, f)